
from app.services.rag_chain import answer_question


def main():

    print("\n======================================")
    print("   RETRIEVAL AUGMENTED GENERATION")
    print("======================================")

    print("Ask questions about our company policies.")

    print("Type 'exit' to stop the chatbot.")

    while True:

        question = input("\nCustomer: ").strip()

        if question.lower() in ["exit", "quit", "bye"]:
            print("\nSession ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            answer, sources = answer_question(question)

            print("\nResponse:")
            print(answer)

            if sources:

                print("\nRetrieved document references:")

                for source in sources:
                    print("-", source)

        except Exception as error:

            print("\nSomething went wrong.")
            print("Error:", error)


if __name__ == "__main__":
    main()
