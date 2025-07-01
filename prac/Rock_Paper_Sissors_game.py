import random

choices = ["rock", "paper", "scissors"]
player = None
computer = random.choice(choices)

while player not in choices:
    player = input("what is your move ").lower()


if player == computer:
    print("it is a tie")
    print("computer:", computer)
    print("player:", player)
elif player == "rock":
    if computer == "paper":
        print("u looose")
        print("computer:", computer)
        print("player:", player)
    if computer == "scissors":
        print("we win")
        print("computer:", computer)
        print("player:", player)
elif player == "scissors":
    if computer == "paper":
        print("we win")
        print("computer:", computer)
        print("player:", player)
    if computer == "rock":
        print("u looose")
        print("computer:", computer)
        print("player:", player)
elif player == "paper":
    if computer == "scissors":
        print("u looose")
        print("computer:", computer)
        print("player:", player)
    if computer == "rock":
        print("we win")
        print("computer:", computer)
        print("player:", player)
