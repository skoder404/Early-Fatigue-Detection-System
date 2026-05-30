from twilio.rest import Client

def send_whatsapp_alert(account_sid, auth_token, from_number, to_numbers, message_text):
    client = Client(account_sid, auth_token)

    sent_sids = []
    for number in to_numbers:
        msg = client.messages.create(
            body=message_text,
            from_=from_number,
            to=number
        )
        sent_sids.append(msg.sid)

    return sent_sids