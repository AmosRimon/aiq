# -----------------------------------------------------------------------------
# Deepchecks LLM Onboarding Script
# -----------------------------------------------------------------------------
# This script is designed to guide you through the process of integrating
# data with Deepchecks LLM Product, using Deepchecks' SDK.
# Please follow the instructions provided and replace placeholders with actual
# data as needed.


# Step 1: Import necessary modules
# Create new venv and install deepchecks' SDK using `pip install deepchecks-llm-client`
import uuid
from datetime import datetime

from deepchecks_llm_client.client import DeepchecksLLMClient
from deepchecks_llm_client.data_types import AnnotationType, EnvType, LogInteraction, Step, UserValueProperty


def main(dc_sdk_client: DeepchecksLLMClient):
    # pylint: disable=invalid-name

    # Use "Update Samples" in deepchecks' service, to create a new application name and place it here
    DEEPCHECKS_APP_NAME = "DemoApp"

    # Step 3: Initialize the SDK
    # Notice - deepchecks SDK was designed to be non-intrusive. It does not throw exceptions
    # and only print to log. By default, it prints only errors from the "init" phase
    # If you wish to increase verbosity, use `verbose=True` and `log_level=logging.INFO`
    version = "0.0.1"
    env_type = EnvType.EVAL

    print(f"app: {DEEPCHECKS_APP_NAME}, version: {version}, env: {env_type}")

    # Step 4: Explicit Logging of data and optionally addition of steps for flow visibility
    # Notice that we can also log user value properties - you can use user value properties that were
    # already defined via the UI, or use the "CP_xxx" convention to create new user value property
    # on the fly

    # We can explicitly log interaction
    print("log interaction using explicit Deepchecks API: log_interaction()")
    user_inter_id = str(uuid.uuid4())
    dc_sdk_client.log_interaction(
        app_name=DEEPCHECKS_APP_NAME,
        version_name=version,
        env_type=env_type,
        interaction=LogInteraction(
            input="my user input",
            output="my model output",
            expected_output="my expected_output",
            full_prompt="system part: my user input",
            annotation=AnnotationType.BAD,
            user_interaction_id=user_inter_id,
            started_at=datetime(2021, 10, 31, 15, 1, 0).astimezone(),
            finished_at=datetime.utcnow().astimezone(),
            steps=[
                Step(
                    name="my_logic_1",
                    value="system part: my information retrieval"
                ),
                Step(
                    value="my user input + the retried data",
                    name="my_logic_2",
                ),
                Step(
                    name="my_logic_3",
                    value="my model output after PII removal"
                )
            ],
            user_value_properties=[UserValueProperty(name="CP_MY_PROP1", value=0.1)],
            topic="Some Topic",
        )
    )

    # Notice that we can re-annotate our logged interation in a later phase
    # using out own external id
    print(f"Annotating explicit interaction with key: {user_inter_id}")
    dc_sdk_client.annotate(
        app_name=DEEPCHECKS_APP_NAME,
        user_interaction_id=user_inter_id,
        version_name=version,
        annotation=AnnotationType.GOOD,
    )

    # Send batch of interactions to
    dc_sdk_client.log_batch_interactions(
        app_name=DEEPCHECKS_APP_NAME,
        version_name=version,
        env_type=env_type,
        interactions=[
            LogInteraction(
                input="my user input2",
                output="my model output2",
                expected_output="my expected_output2",
                full_prompt="system part: my user input2",
                annotation=AnnotationType.GOOD,
                user_interaction_id=str(uuid.uuid4()),
                started_at=datetime(2021, 10, 31, 15, 1, 0).astimezone(),
                finished_at=datetime.utcnow().astimezone(),
                steps=[
                    Step(
                        name="Information Retrieval",
                        value="system part: my information retrieval"),
                ],
                user_value_properties=[UserValueProperty(name="CP_MY_PROP1", value=0.1)]
            ),
            LogInteraction(
                input="my user input3",
                output="my model output3",
                expected_output="my expected_output3",
                full_prompt="system part: my user input3",
                annotation=AnnotationType.GOOD,
                user_interaction_id=str(uuid.uuid4()),
                started_at=datetime(2021, 10, 31, 15, 1, 0).astimezone(),
                finished_at=datetime.utcnow().astimezone(),
                user_value_properties=[UserValueProperty(name="CP_MY_PROP1", value=0.1)]
            ),
        ]
    )

    # Step 5: Creating a new version based on the "0.0.1" eval set

    # Set different version / env type, to be carried on to the next calls
    # Usually, in "dev mode", you will log to "EnvType.EVAL" and use a different version
    # for each evaluation you make. Check the performance of your LLM application using deepchecks,
    # and deploy the selected version to production, and from that point on,
    # you will use "EnvType.PROD" with the "production version" to log production data
    new_version = "0.0.2"
    eval_env_type = EnvType.EVAL
    print(f"switching to version: {new_version}, and EnvType: {eval_env_type}")

    # Log interaction for V2
    dc_sdk_client.log_interaction(
        app_name=DEEPCHECKS_APP_NAME,
        version_name=new_version,
        env_type=eval_env_type,
        interaction=LogInteraction(
            input="my user input",
            output="my model output",
            expected_output="my expected_output",
            full_prompt="system part: my user input",
            annotation=AnnotationType.BAD,
            user_interaction_id=user_inter_id,
            started_at=datetime(2021, 10, 31, 15, 1, 0).astimezone(),
            finished_at=datetime.utcnow().astimezone(),
            steps=[],
            user_value_properties=[UserValueProperty(name="CP_MY_PROP2", value=1)],
            topic="Some Topic",
        )
    )

    # Now we can also retrieve eval set as dataframe, for example:
    df = dc_sdk_client.get_data(app_name=DEEPCHECKS_APP_NAME, version_name="0.0.1", env_type=EnvType.EVAL)
    print("We can also fetch the eval set as dataframe, here are the dataframe columns")
    print(list(df.columns))


if __name__ == "__main__":
    # Set up configuration

    # Fill deepchecks host name here
    DEEPCHECKS_LLM_HOST = "https://host-name-here"  # could be: https://app.llm.deepchecks.com

    # Login to deepchecks' service and generate new API Key (Configuration -> API Key) and place it here
    DEEPCHECKS_LLM_API_KEY = "Fill Key Here"

    print(f"going to init deepchecks client, host: {DEEPCHECKS_LLM_HOST}")

    client = DeepchecksLLMClient(host=DEEPCHECKS_LLM_HOST, api_token=DEEPCHECKS_LLM_API_KEY)
    main(client)
