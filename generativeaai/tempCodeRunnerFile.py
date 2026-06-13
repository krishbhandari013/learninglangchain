

        try:
            print("\nAI: ", end="", flush=True)
            for chunk in agent.stream(
                {"messages": [("user", query)]},