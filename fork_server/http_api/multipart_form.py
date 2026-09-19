# coding: utf-8
"""Cancellation-safe ownership wrapper around Starlette multipart parsing."""

from __future__ import annotations

from contextlib import suppress

from starlette.formparsers import MultiPartParser


async def parse_multipart_form(
    request,
    *,
    max_files: int,
    max_fields: int,
):
    """Parse a form and close partial spools on every exceptional exit.

    Starlette's parser closes ``_files_to_close_on_error`` only for
    ``MultiPartException``.  Client disconnects and task cancellation can occur
    after a file has been created but before ``Request.form().__aenter__``
    returns, so the caller never receives a context manager to close.  Owning
    the parser here makes that cleanup unconditional.
    """
    parser = MultiPartParser(
        request.headers,
        request.stream(),
        max_files=max_files,
        max_fields=max_fields,
    )
    try:
        return await parser.parse()
    except BaseException:
        for file in tuple(getattr(parser, "_files_to_close_on_error", ())):
            with suppress(Exception):
                file.close()
        raise


def close_form_files(form) -> None:
    """Synchronously close every completed upload without a cancellation point."""
    for _field, value in form.multi_items():
        file = getattr(value, "file", None)
        if file is not None:
            with suppress(Exception):
                file.close()
