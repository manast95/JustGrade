# Inject system trust store (macOS Keychain, incl. Zscaler root CA) into Python's ssl.
# Must run before any network library (httpx, requests) is imported.
# truststore reads the system keychain without requiring sudo.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    # Fall back to certifi-based patch for non-enterprise setups
    try:
        import ssl, certifi
        _orig = ssl.create_default_context
        def _certifi_ctx(purpose=ssl.Purpose.SERVER_AUTH, *,
                         cafile=None, capath=None, cadata=None):
            if cafile is None and capath is None and cadata is None:
                cafile = certifi.where()
            return _orig(purpose, cafile=cafile, capath=capath, cadata=cadata)
        ssl.create_default_context = _certifi_ctx
    except (ImportError, AttributeError):
        pass
