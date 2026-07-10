import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

try:
    run = client.create_run(
        name="test_run",
        run_type="chain",
        project_name=os.getenv("LANGCHAIN_PROJECT"),
        inputs={"question": "Hello"},
        outputs={"answer": "World"},
    )
    print("SUCCESS")
    print(run)
except Exception as e:
    import traceback
    traceback.print_exc()