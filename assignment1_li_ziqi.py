# Name: Li Ziqi
# Assignment One

import turtle


def turtle_drawing():
    screen = turtle.Screen()
    screen.title("Li Ziqi - Turtle Drawing")
    screen.bgcolor("black")
    pen = turtle.Turtle()
    pen.speed(0)
    colors = ["red", "orange", "yellow", "green", "blue", "purple"]

    # Draw colourful squares with changing colours and side lengths.
    for i in range(36):
        pen.pencolor(colors[i % len(colors)])
        length = 40 + i * 4
        for side in range(4):
            pen.forward(length)
            pen.right(90)
        pen.right(10)

    pen.hideturtle()
    screen.exitonclick()


def qa_bot():
    print("Question Answering Bot")
    print("Ask about HKU, robot, jetson, ai, or human. Type quit to finish.")

    while True:
        question = input("Ask me something: ").strip().lower()
        if question == "quit":
            print("Bot: Goodbye!")
            break
        elif "hku" in question:
            print("Bot: HKU stands for The University of Hong Kong.")
        elif "robot" in question:
            print("Bot: A robot is a machine that can carry out tasks automatically.")
        elif "jetson" in question:
            print("Bot: Jetson Nano is an AI computer made by NVIDIA.")
        elif "ai" in question:
            print("Bot: AI means Artificial Intelligence.")
        elif "human" in question:
            print("Bot: A human is a person who can think, learn, and feel.")
        else:
            print("Bot: Sorry, I don't understand.")


def simple_calculator():
    print("Simple Calculator")
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
    op = input("Choose operation (+, -, *, /): ").strip()

    if op == "+":
        print("Result:", num1 + num2)
    elif op == "-":
        print("Result:", num1 - num2)
    elif op == "*":
        print("Result:", num1 * num2)
    elif op == "/":
        if num2 == 0:
            print("Cannot divide by zero.")
        else:
            print("Result:", num1 / num2)
    else:
        print("Invalid operation")


if __name__ == "__main__":
    turtle_drawing()
    qa_bot()
    simple_calculator()
