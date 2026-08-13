import boto3

from app.config import settings

_client = None


def get_s3_client():
    global _client
    if _client is None:
        kwargs = {"region_name": settings.aws_region}
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        _client = boto3.client("s3", **kwargs)
    return _client
