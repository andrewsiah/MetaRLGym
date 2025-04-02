import textarena as ta
from textarena.agents.basic_agents import OpenRouterAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded
if "OPENROUTER_API_KEY" not in os.environ:
    raise ValueError("Please ensure OPENROUTER_API_KEY is set in your .env file")

# # Test if OpenRouterAgent is working properly
# def test_openrouter_agent():
#     print("Testing OpenRouterAgent...")
#     agent = OpenRouterAgent(
#         model_name="mistral/ministral-8b",
#         verbose=True,
#     )
#     test_prompt = "Hello! Can you respond with a short greeting? This is a test to check if you're working properly."
#     print(f"Sending test prompt: {test_prompt}")
#     try:
#         response = agent(test_prompt)
#         print(f"Received response: {response}")
#         print("OpenRouterAgent test successful!")
#         return True
#     except Exception as e:
#         print(f"OpenRouterAgent test failed with error: {e}")
#         return False

# # Run the test before starting the game
# if not test_openrouter_agent():
#     print("Exiting due to OpenRouterAgent test failure")
#     exit(1)

# Initialize agents
agents = {
    0: OpenRouterAgent(
        model_name="meta-llama/llama-3.2-1b-instruct",
        verbose=True,
    ),
    1: OpenRouterAgent(
        model_name="meta-llama/llama-3.2-1b-instruct",
        verbose=True,
    ),
}

# Initialize environment
env = ta.make("Stratego-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "llama-3.2-1b-instruct-1", 1: "llama-3.2-1b-instruct-2"},
)
env.reset(num_players=2)
done = False

# Run the simulation
while not done:
    player_id, observation = env.get_observation()
    # print(f"Player {player_id} ({agents[player_id].model_name}) observation:")
    # print(observation)
    action = agents[player_id](observation)
    # print(f"Player {player_id} action:")
    # print(action)
    done, info = env.step(action=action)
    # print(f"Done: {done}, Info: {info}")
    print("-" * 50)

env.close()
print("Game Over!")