"""The call's tool rules, free of Pipecat, so bot.py and the prompt eval (scripts/eval_prompt.py) run the same code.
Each method returns (result for the model, text the system speaks or None)."""
from agent.prompts import DISPOSITIONS, GOODBYE, reminder_text
from agent.ptp import validate_promise
from agent.record import CUSTOMER, identity_result

IDENTITY_ATTEMPTS = 3
VERIFIED_ONLY = {'already_paid', 'promise_to_pay', 'refused_to_pay', 'loan_not_recognised'}


class CallFlow:
    def __init__(self, lang, today):
        self.lang, self.today = lang, today
        self.verified, self.id_attempts, self.ptp = False, 0, None

    def verify_identity(self, args, heard):
        """heard: the caller's last transcript, which must back up the model's reading of it."""
        self.id_attempts += 1
        outcome = identity_result(args.get('answer'), args.get('name_said'), heard)
        if outcome == 'verified':
            self.verified = True
            return {'verified': True, 'reminder_spoken': True}, reminder_text(self.lang)
        return {'verified': False, 'reason': outcome,
                'attempts_left': max(0, IDENTITY_ATTEMPTS - self.id_attempts)}, None

    def repeat_reminder(self, args):
        if not self.verified:
            return {'ok': False, 'error': 'identity_not_verified'}, None
        return {'ok': True, 'reminder_spoken': True}, reminder_text(self.lang)

    def record_promise_to_pay(self, args):
        if not self.verified:
            return {'ok': False, 'error': 'identity_not_verified'}, None
        r = validate_promise(args.get('date', ''), self.today, CUSTOMER['emi_amount_inr'], CUSTOMER['loan_id'])
        if r['ok']:
            self.ptp = r
            return {'ok': True, 'reference_id': r['reference_id'], 'confirmation_spoken': True}, r['confirmation'][self.lang]
        return {'ok': False, 'error': r['error'], 'asked_for_another_date': True}, r['say_text'][self.lang]

    def end_call(self, args):
        """Also returns the disposition details to log; the spoken text is the scripted goodbye."""
        d = args.get('disposition')
        error = ('unknown disposition' if d not in DISPOSITIONS else
                 'identity not verified yet' if d in VERIFIED_ONLY and not self.verified else
                 'no promise recorded; call record_promise_to_pay first' if d == 'promise_to_pay' and not self.ptp else
                 'ask for a callback time first' if d == 'callback_requested' and not args.get('callback_time') else None)
        if error:
            return {'ok': False, 'error': error}, None, None
        details = ({'promise_date': self.ptp['date_iso'], 'reference_id': self.ptp['reference_id']} if d == 'promise_to_pay'
                   else {'callback_time': args['callback_time']} if d == 'callback_requested' else {})
        return {'ok': True}, GOODBYE[d][self.lang], details
