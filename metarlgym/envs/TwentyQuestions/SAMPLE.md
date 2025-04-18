# Sample Run

This file demonstrates a sample full run of the Twenty Questions environment (Basic).

## Initial Observation
On reset, the environment returns:
```plaintext
[GAME] You are Player 0. You are playing 20 Questions (Basic).
The gamemaster has chosen an object that can be one or two words. This object is related to places. You have to guess this object by asking yes-or-no questions.
The game will last for a maximum of 20 questions. After 20 questions, the gamemaster will prompt you to make a guess.
You may ask your question in any manner, so long they are not wrapped in square brackets.
Then, to make your final word guess, ensure that you wrap it with square brackets, e.g. [plane], [diving bell].
As you play, the history of your questions and gamemaster's responses will be displayed.
```

## Sample Interaction
Below is a sequence of questions and responses leading to the correct guess of "elephant".
```plaintext
[Player 0] Is it a living thing?
[GAME] Yes

[Player 0] Is it an animal?
[GAME] Yes

[Player 0] Is it a mammal?
[GAME] Yes

[Player 0] Does it live in water?
[GAME] No

[Player 0] Is it domesticated?
[GAME] No

[Player 0] Is it larger than a breadbox?
[GAME] Yes

[Player 0] Does it have tusks?
[GAME] Yes

[Player 0] Is it native to Africa?
[GAME] Yes

[Player 0] [elephant]
[GAME] Congratulations! You guessed the object correctly. The word was 'elephant'.
```
