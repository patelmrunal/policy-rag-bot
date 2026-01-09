import os
from rag_pipeline import qa_bot


def main():
    print("--- Initializing Policy Bot ---")
    qa = qa_bot()

    if not qa:
        print("Initialization failed.")
        return

    print("Bot is ready! Type 'exit' to stop.")
    print("------------------------------------------------")

    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        res = qa.invoke({"input": query})
        answer = res["answer"]
        source_docs = res["context"]

        print(f"\nBot: {answer}")

        #  Improved Source Display
        print("\n[Sources Used]")
        seen_pages = set()

        for doc in source_docs:
            source_file = os.path.basename(doc.metadata.get("source", "Unknown File"))
            page_num = doc.metadata.get("page", "?")

            source_id = f"{source_file} (Page {page_num})"

            if source_id not in seen_pages:
                print(f"- {source_id}")
                seen_pages.add(source_id)

        # Print a small separator for clarity
        print("------------------------------------------------")


if __name__ == "__main__":
    main()
