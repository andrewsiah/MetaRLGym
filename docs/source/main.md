    def __init__(
        self, host: str = "0.0.0.0", server_port: int = 8000, group_port: int = 51217, connection_timeout: float = 0.0
    ):

    There's a bug where our VLM keeps saying that our group pod is occupied no matter how we kill and restart it doesn't work. Going into VLM client and changing the group pod manually works. 