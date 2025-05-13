def fetch_token_data(address: str) -> dict:
    logger.info(f"Fetching data for address: {address}")
    network_id = 101 if len(address) > 40 else 1
    query = """
    query GetTokenInfo($input: TokenInput!) {
        token(input: $input) {
            address
            name
            totalSupply
            info {
                circulatingSupply
            }
        }
    }
    """
    body = {
        "query": query,
        "variables": {
            "input": {
                "address": address,
                "networkId": network_id
            }
        }
    }
    headers = {
        "Authorization": f"Bearer {CODEX_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(CODEX_API_URL, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Codex.io response: {data}")
        if "errors" in data:
            logger.error(f"Codex.io API error: {data['errors']}")
            return {"no_data": True, "message": "That token doesn’t deserve a reply"}
        if not data.get("data") or not data["data"].get("token"):
            logger.info(f"No token data for address: {address}")
            return {"no_data": True, "message": "That token doesn’t deserve a reply"}
        token = data["data"]["token"]
        return {
            "address": token["address"],
            "name": token["name"],
            "totalSupply": str(token["totalSupply"]),
            "circulatingSupply": str(token["info"]["circulatingSupply"] or "0")
        }
    except requests.RequestException as e:
        logger.error(f"Codex.io request error for address {address}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch token data: {str(e)}")

def generate_reply(token_data: dict, query: str, user: str) -> str:
    if token_data.get("no_data"):
        return token_data["message"]
    system = (
        "You’re a degenerate crypto gambler. Keep it ≤240 chars, "
        "snarky but useful, mocking pumps and rugs, include one data point."
    )
    user_msg = (
        f"User asked about {query}. "
        f"Supply={token_data['totalSupply']}, circ={token_data['circulatingSupply']}. "
        "Give a punchy degen take."
    )
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": 200,
        "temperature": 0.9
    }
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text[:240]
    except requests.RequestException as e:
        logger.error(f"Grok API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Grok analysis failed: {str(e)}")