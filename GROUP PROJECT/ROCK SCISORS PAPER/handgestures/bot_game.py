import os
import telebot
import random

API_KEY = os.getenv('API_KEY')
bot = telebot.TeleBot(API_KEY)

choices = ['rock', 'paper', 'scissors']
computer_choice = random.choice(choices)


@bot.message_handler(commands=['play'])
def start(message):
   bot.send_message(message.chat.id, '''select from the following
   -rock 
   -paper
   -scisors ''')

def user_input(message):
  request = message.text.lower()
  if request not in ['rock', 'paper', 'scissors']:
    return False
  else:
    return True


@bot.message_handler(func=user_input)
def send_output(message):
  player_choice = message.text.lower()

  computer_choice = random.choice(choices)

  if player_choice == computer_choice:
    bot.send_message(message.chat.id, "It's a Tie")      
  elif player_choice == 'rock' and computer_choice == 'scissors':
      bot.send_message(message.chat.id, "Player wins!")
  elif player_choice == 'scissors' and computer_choice == 'paper':
      bot.send_message(message.chat.id, "Player wins!")
  elif player_choice == 'paper' and computer_choice == 'rock':
      bot.send_message(message.chat.id, "Player wins!")
  else:
      bot.send_message(message.chat.id, "Computer wins!")
      bot.send_message(message.chat.id,"computer picked: %s" % computer_choice)  
    




bot.polling()