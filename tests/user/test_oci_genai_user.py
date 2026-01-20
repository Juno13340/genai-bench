from unittest.mock import MagicMock, patch

import pytest
from oci.generative_ai_inference.models import (
    ChatDetails,
    DedicatedServingMode,
    GenericChatRequest,
    OnDemandServingMode,
)

from genai_bench.protocol import UserChatRequest, UserEmbeddingRequest
from genai_bench.user.oci_genai_user import OCIGenAIUser


@pytest.fixture
def test_genai_user():
    OCIGenAIUser.host = "http://example.com"

    # Set up mock auth provider
    mock_auth = MagicMock()
    mock_auth.get_credentials.return_value = "test-key"
    mock_auth.get_config.return_value = {
        "api_base": "http://example.com",
        "compartment_id": "test-compartment",
        "endpoint_id": "test-endpoint",
    }
    OCIGenAIUser.auth_provider = mock_auth

    user = OCIGenAIUser(environment=MagicMock())
    return user


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_grok_format(mock_client_class, test_genai_user):
    """Test chat with Grok format streaming response."""
    # Mock the chat method
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    # Prepare test streaming response data
    hello_msg = (
        '{"message": {"role": "ASSISTANT", '
        '"content": [{"type": "TEXT", "text": "Hello"}]}}'
    )
    world_msg = (
        '{"message": {"role": "ASSISTANT", '
        '"content": [{"type": "TEXT", "text": " world"}]}}'
    )
    exclamation_msg = (
        '{"message": {"role": "ASSISTANT", '
        '"content": [{"type": "TEXT", "text": "!"}]}}'
    )

    usage_msg = (
        '{"usage": {"totalTokens": 8, "promptTokens": 5, '
        '"completionTokens": 3, "completionTokensDetails": {"reasoningTokens": 0}}}'
    )

    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=hello_msg),
            MagicMock(data=world_msg),
            MagicMock(data=exclamation_msg),
            MagicMock(data='{"finishReason": "stop"}'),
            MagicMock(data=usage_msg),
        ]
    )

    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserChatRequest(
        model="xai.grok-3-mini-fast",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_genai_user.collect_metrics = metrics_collector_mock

    test_genai_user.chat()

    # Verify the correct request was made
    mock_client_instance.chat.assert_called_once_with(
        ChatDetails(
            compartment_id="ocid1.compartment.oc1..example",
            serving_mode=OnDemandServingMode(model_id="xai.grok-3-mini-fast"),
            chat_request=GenericChatRequest(
                api_format="GENERIC",
                messages=[
                    {"role": "USER", "content": [{"text": "Hello", "type": "TEXT"}]}
                ],
                max_tokens=10,
                is_stream=True,
                stream_options={"isIncludeUsage": True},
                temperature=0.75,
                top_p=0.7,
                top_k=1,
                frequency_penalty=None,
                presence_penalty=None,
                num_generations=1,
            ),
        )
    )

    # Verify metrics were collected
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 5
    assert user_response.generated_text == "Hello world!"
    assert user_response.tokens_received == 3  # totalTokens (8) - promptTokens (5)


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_with_system_message(mock_client_class, test_genai_user):
    """Test chat with system message in additional params."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = []

    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserChatRequest(
        model="xai.grok-3-mini-fast",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
            "system_message": "You are a helpful assistant.",
        },
        max_tokens=10,
    )

    test_genai_user.chat()

    chat_request = mock_client_instance.chat.call_args[0][0].chat_request
    expected_messages = [
        {
            "role": "SYSTEM",
            "content": [{"text": "You are a helpful assistant.", "type": "TEXT"}],
        },
        {"role": "USER", "content": [{"text": "Hello", "type": "TEXT"}]},
    ]
    assert chat_request.messages == expected_messages


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_with_chat_history(mock_client_class, test_genai_user):
    """Test chat with chat history in additional params."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = []

    test_genai_user.on_start()
    chat_history = [
        {"role": "user", "content": "Previous message"},
        {"role": "assistant", "content": "Previous response"},
    ]
    test_genai_user.sample = lambda: UserChatRequest(
        model="xai.grok-3-mini-fast",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
            "chat_history": chat_history,
        },
        max_tokens=10,
    )

    test_genai_user.chat()

    chat_request = mock_client_instance.chat.call_args[0][0].chat_request
    expected_messages = [
        {"role": "USER", "content": [{"text": "Previous message", "type": "TEXT"}]},
        {
            "role": "ASSISTANT",
            "content": [{"text": "Previous response", "type": "TEXT"}],
        },
        {"role": "USER", "content": [{"text": "Hello", "type": "TEXT"}]},
    ]
    assert chat_request.messages == expected_messages


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_dedicated_serving_mode(mock_client_class, test_genai_user):
    """Test chat with DEDICATED serving mode."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = []

    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserChatRequest(
        model="xai.grok-3-mini-fast",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "DEDICATED",
            "endpointId": "ocid1.endpoint.oc1..example",
        },
        max_tokens=10,
    )

    test_genai_user.chat()

    chat_detail = mock_client_instance.chat.call_args[0][0]
    assert isinstance(chat_detail.serving_mode, DedicatedServingMode)
    assert chat_detail.serving_mode.endpoint_id == "ocid1.endpoint.oc1..example"


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_missing_endpoint_id_for_dedicated(mock_client_class, test_genai_user):
    """Test error when endpointId is missing for DEDICATED serving mode."""
    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserChatRequest(
        model="xai.grok-3-mini-fast",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "DEDICATED",
            # Missing endpointId
        },
        max_tokens=10,
    )

    with pytest.raises(
        ValueError, match="endpointId must be provided for DEDICATED servingType"
    ):
        test_genai_user.chat()


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_missing_compartment_id(mock_client_class, test_genai_user):
    """Test error when compartmentId is missing."""
    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserChatRequest(
        model="xai.grok-3-mini-fast",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "servingType": "ON_DEMAND",
            # Missing compartmentId
        },
        max_tokens=10,
    )

    with pytest.raises(
        ValueError, match="compartmentId missing in additional request params"
    ):
        test_genai_user.chat()


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_with_wrong_request_type(mock_client_class, test_genai_user):
    """Test error when wrong request type is provided."""
    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserEmbeddingRequest(
        documents=[],
        num_prefill_tokens=10,
        model="xai.grok-3-mini-fast",
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    with pytest.raises(AttributeError):
        test_genai_user.chat()


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_with_response_error(mock_client_class, test_genai_user):
    """Test handling of HTTP error responses."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 401
    mock_client_instance.chat.return_value.request_id = "test-request-id"
    mock_client_instance.chat.return_value.response = "Unauthorized"

    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserChatRequest(
        model="xai.grok-3-mini-fast",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_genai_user.collect_metrics = metrics_collector_mock

    response = test_genai_user.chat()

    assert response.status_code == 401
    assert response.error_message == "Request Failed"
    metrics_collector_mock.assert_called_once()


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_json_decode_error(mock_client_class, test_genai_user):
    """Test handling of JSON decode errors in streaming response."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    # Prepare test data with JSON decode error
    hello_response = (
        '{"message": {"role": "ASSISTANT", '
        '"content": [{"type": "TEXT", "text": "Hello"}]}}'
    )
    world_response = (
        '{"message": {"role": "ASSISTANT", '
        '"content": [{"type": "TEXT", "text": " world"}]}}'
    )

    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=hello_response),
            MagicMock(data="invalid json"),  # This should cause JSON decode error
            MagicMock(data=world_response),
            MagicMock(data='{"finishReason": "stop"}'),
        ]
    )

    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserChatRequest(
        model="xai.grok-3-mini-fast",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_genai_user.collect_metrics = metrics_collector_mock

    # Should handle the JSON error gracefully
    response = test_genai_user.chat()

    assert response.status_code == 200
    assert "Hello world" in response.generated_text


def test_build_messages_with_all_components(test_genai_user):
    """Test build_messages with system message and chat history."""
    user_request = UserChatRequest(
        model="xai.grok-3-mini-fast",
        prompt="Current message",
        num_prefill_tokens=5,
        additional_request_params={
            "system_message": "You are helpful.",
            "chat_history": [
                {"role": "user", "content": "Previous user message"},
                {"role": "assistant", "content": "Previous assistant message"},
            ],
        },
        max_tokens=10,
    )

    messages = test_genai_user.build_messages(user_request)

    expected_messages = [
        {"role": "SYSTEM", "content": [{"text": "You are helpful.", "type": "TEXT"}]},
        {
            "role": "USER",
            "content": [{"text": "Previous user message", "type": "TEXT"}],
        },
        {
            "role": "ASSISTANT",
            "content": [{"text": "Previous assistant message", "type": "TEXT"}],
        },
        {"role": "USER", "content": [{"text": "Current message", "type": "TEXT"}]},
    ]

    assert messages == expected_messages


def test_build_messages_minimal(test_genai_user):
    """Test build_messages with only prompt."""
    user_request = UserChatRequest(
        model="xai.grok-3-mini-fast",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={},
        max_tokens=10,
    )

    messages = test_genai_user.build_messages(user_request)

    expected_messages = [
        {"role": "USER", "content": [{"text": "Hello", "type": "TEXT"}]}
    ]
    assert messages == expected_messages


def test_backend_name():
    """Test backend name constant."""
    assert OCIGenAIUser.BACKEND_NAME == "oci-genai"


def test_supported_tasks():
    """Test supported tasks mapping."""
    assert OCIGenAIUser.supported_tasks == {
        "text-to-text": "chat",
    }


def test_on_start_without_auth():
    """Test on_start raises error without auth provider."""
    user = OCIGenAIUser(environment=MagicMock())
    user.auth_provider = None

    with pytest.raises(ValueError, match="Auth is required for OCIGenAIUser"):
        user.on_start()


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_with_reasoning_content(mock_client_class, test_genai_user):
    """Test chat with reasoning_content in streaming response."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200

    # Prepare test streaming response with reasoning_content
    reasoning_msg = (
        '{"message": {"role": "ASSISTANT", '
        '"reasoning_content": [{"type": "TEXT", "text": "Let me think..."}]}}'
    )
    content_msg = (
        '{"message": {"role": "ASSISTANT", '
        '"content": [{"type": "TEXT", "text": "The answer is 42"}]}}'
    )
    usage_msg = (
        '{"usage": {"totalTokens": 20, "promptTokens": 5, '
        '"completionTokens": 10, "completionTokensDetails": {"reasoningTokens": 5}}}'
    )

    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=reasoning_msg),
            MagicMock(data=content_msg),
            MagicMock(data='{"finishReason": "stop"}'),
            MagicMock(data=usage_msg),
        ]
    )

    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserChatRequest(
        model="openai.gpt-oss-120b",
        prompt="What is the answer?",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
            "reasoning_effort": "MEDIUM",
        },
        max_tokens=20,
    )

    metrics_collector_mock = MagicMock()
    test_genai_user.collect_metrics = metrics_collector_mock

    response = test_genai_user.chat()

    # Verify the response includes both reasoning and regular content
    assert response.status_code == 200
    assert response.generated_text == "Let me think...The answer is 42"
    # Verify completionTokens is used (10), not totalTokens - promptTokens (15)
    assert response.tokens_received == 10
    assert response.time_at_first_token is not None


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_reasoning_effort_parameter(mock_client_class, test_genai_user):
    """Test that reasoning_effort parameter is passed to the API."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter([])

    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserChatRequest(
        model="openai.gpt-oss-120b",
        prompt="Test prompt",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
            "reasoning_effort": "HIGH",
        },
        max_tokens=10,
    )

    test_genai_user.chat()

    # Verify reasoning_effort was included in the request
    chat_request = mock_client_instance.chat.call_args[0][0].chat_request
    assert chat_request.reasoning_effort == "HIGH"


@patch("genai_bench.user.oci_genai_user.GenerativeAiInferenceClient")
def test_chat_ttft_triggered_by_reasoning_content(mock_client_class, test_genai_user):
    """Test that TTFT is triggered by reasoning_content (first token)."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200

    # Only reasoning_content, no regular content
    reasoning_msg = (
        '{"message": {"role": "ASSISTANT", '
        '"reasoning_content": [{"type": "TEXT", "text": "Thinking..."}]}}'
    )
    usage_msg = (
        '{"usage": {"totalTokens": 10, "promptTokens": 5, '
        '"completionTokens": 3, "completionTokensDetails": {"reasoningTokens": 2}}}'
    )

    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=reasoning_msg),
            MagicMock(data='{"finishReason": "stop"}'),
            MagicMock(data=usage_msg),
        ]
    )

    test_genai_user.on_start()
    test_genai_user.sample = lambda: UserChatRequest(
        model="openai.gpt-oss-120b",
        prompt="Think about this",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_genai_user.collect_metrics = metrics_collector_mock

    response = test_genai_user.chat()

    # Verify TTFT was triggered by reasoning_content
    assert response.time_at_first_token is not None
    assert response.generated_text == "Thinking..."
    # Verify completionTokens (3) is used, not totalTokens - promptTokens (5)
    assert response.tokens_received == 3
