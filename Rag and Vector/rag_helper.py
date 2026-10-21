INSTRUCTIONS = """
You are a helpful Currency FAQ AI Agent.

Answer questions using ONLY the provided context.

Rules:
- Do not invent information.
- If the answer is not in the context, respond with "I don't know."
- Do not provide exchange rates because the dataset does not contain them.
- Keep answers concise and accurate.
""".strip()


PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="openai/gpt-oss-120b"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {
            "code": 3.0,
            "name": 3.0,
            "symbol": 2.0,
            "countries": 2.0
        }

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append("Currency code: " + str(doc["code"]))
            lines.append("Currency name: " + str(doc["name"]))
            lines.append("Symbol: " + str(doc["symbol"]))
            lines.append(
                "Decimal digits: " + str(doc["decimal_digits"])
            )
            lines.append("Countries: " + str(doc["countries"]))
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)

        return self.prompt_template.format(
            question=query,
            context=context
        )

    def llm(self, prompt):
        input_messages = [
            {
                "role": "developer",
                "content": self.instructions
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    def rag(self, query):
        search_results = self.search(query)

        if not search_results:
            return "I don't know."

        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)

        return answer