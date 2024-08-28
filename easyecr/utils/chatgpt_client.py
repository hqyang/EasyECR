"""
https://openai.com/blog/chatgpt
https://platform.openai.com/docs/guides/chat
"""
import time
import argparse
import json
import traceback

from openai import OpenAI


def chat(client: OpenAI, **kwargs):
    """

    Args:
        client:
        **kwargs:

    Returns:

    """
    response = client.chat.completions.create(**kwargs)
    return response


def simple_chat(messages, client: OpenAI, model: str = 'gpt-3.5-turbo-0613', max_tokens: int = 200, request_timeout: int = 10,
                max_try_num: int = 3):
    """

    Args:
        messages:
        client:
        model:
        max_tokens:
        request_timeout:
        max_try_num:

    Returns:

    """
    data = {
        "model": model,
        "messages": messages,
        'max_tokens': max_tokens,
        'timeout': request_timeout,
        'temperature': 0.0
    }

    try_num = 0
    while try_num < max_try_num:
        try_num += 1
        try:
            res_obj = chat(client, **data)
            if res_obj.choices[0].finish_reason == 'stop':
                answer = res_obj.choices[0].message.content
                return answer
        except:
            traceback.print_exc()
            time.sleep(3)
    return ''
