"""HTTP client for communicating with the shark-matcher inference service."""

import http.client
import json
import urllib.parse

from core import app_constants
import frontend.common

# Embedding generation blocks for the full pipeline when called synchronously.
_EMBEDDINGS_TIMEOUT = 3600
_DEFAULT_TIMEOUT = 60
_STATUS_TIMEOUT = 10
_HEALTH_TIMEOUT = 5


def health():
    """Check inference-service process liveness."""
    result = _request(
        'GET', '/health', timeout=_HEALTH_TIMEOUT, authenticated=False
    )
    return result is not None and result.get('status') == 'ok'


def get_matches(image_id, retrieval_method=0, model=None):
    """Retrieve nearest-neighbour matches for a query image from the inference service."""
    body = {'image_id': image_id, 'retrieval_method': retrieval_method}
    if model is not None:
        body['model'] = model

    result = _request('POST', '/match', body, required=True)
    if (not isinstance(result, dict)
            or not isinstance(result.get('ids'), list)
            or not isinstance(result.get('distances'), list)
            or len(result['ids']) != len(result['distances'])):
        raise frontend.common.APIError('Invalid matching service response.', 502)
    return result


def get_stats():
    """Fetch retriever state for the dataset-stats panel."""
    return _request('GET', '/stats')


def get_suggestions(review_type='unmatch_suggestions', model=None):
    """Fetch cached unmatch or merge review suggestions from the inference service."""
    body = {'review_type': review_type}
    if model is not None:
        body['model'] = model

    result = _request('POST', '/suggestions', body, timeout=_EMBEDDINGS_TIMEOUT)
    if result is None:
        return []
    return result.get('suggestions', [])


def generate_embeddings(model_name, snapshot_id):
    """Regenerate embeddings for one model from an immutable inference snapshot."""
    body = {
        'model_name': model_name,
        'snapshot_id': snapshot_id,
    }

    result = _request('POST', '/embeddings/generate', body, timeout=_EMBEDDINGS_TIMEOUT)
    return result is not None and result.get('status') == 'ok'


def start_embeddings_generation(model_name, snapshot_id):
    """Start non-blocking background embedding generation for one model."""
    body = {
        'model_name': model_name,
        'snapshot_id': snapshot_id,
        'background': True,
    }
    return _request('POST', '/embeddings/generate', body, timeout=_STATUS_TIMEOUT)


def get_embeddings_status():
    """Fetch the status of the current or most recent embedding generation job."""
    return _request('GET', '/embeddings/status', timeout=_STATUS_TIMEOUT)


def _request(method, path, body=None, timeout=_DEFAULT_TIMEOUT,
             authenticated=True, required=False):
    """Send an HTTP request through the shared transport to the inference service."""
    headers = {}
    if authenticated:
        api_key = app_constants.INFERENCE_API_KEY
        if not api_key:
            print("Inference service credential is not configured; set "
                  f"{app_constants.INFERENCE_API_KEY_ENV} to the matcher's key.")
            if required:
                raise frontend.common.APIError('Matching service is unavailable.', 503)
            return None
        headers['Authorization'] = f"Bearer {api_key}"

    url = urllib.parse.urlparse(app_constants.INFERENCE_SERVICE_URL)
    host = url.hostname
    port = url.port or (443 if url.scheme == 'https' else 80)

    conn = http.client.HTTPConnection(host, port, timeout=timeout)

    try:
        request_kwargs = {}
        if body is not None:
            payload = json.dumps(body).encode('utf-8')
            request_kwargs['body'] = payload
            headers.update({
                'Content-Type': 'application/json',
                'Content-Length': len(payload),
            })
        if headers:
            request_kwargs['headers'] = headers
        conn.request(method, path, **request_kwargs)

        response = conn.getresponse()
        response_body = response.read().decode('utf-8')

        if response.status != 200:
            detail = ""
            try:
                error_payload = json.loads(response_body)
                if isinstance(error_payload, dict) and "detail" in error_payload:
                    detail = f": {error_payload['detail']}"
            except Exception:
                if response_body.strip():
                    detail = f": {response_body.strip()}"
            print(f"Inference service error {response.status} on {method} {path}{detail}")
            if required:
                code = response.status if response.status in (400, 404, 409, 503) else 502
                raise frontend.common.APIError('Matching service could not complete the request.', code)
            return None

        return json.loads(response_body)
    except frontend.common.APIError:
        raise
    except TimeoutError as ex:
        if required:
            raise frontend.common.APIError('Matching service timed out.', 504) from ex
        return None
    except Exception as e:
        print(f"Inference service request failed {method} {path}: {type(e).__name__}")
        if required:
            raise frontend.common.APIError('Matching service is unavailable.', 502) from e
        return None
    finally:
        conn.close()
