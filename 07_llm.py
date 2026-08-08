import subprocess
import json
import getpass


def main():

    api_key ="enter api key"

    query = "What is self-attention?"

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": query
                    }
                ]
            }
        ]
    }

    command = [
        "curl",
        "-s",
        "-X", "POST",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(data)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    response = json.loads(result.stdout)

    print(response)


if __name__ == "__main__":
    main()