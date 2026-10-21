
import time

from tqdm.auto import tqdm
from rag_helper import RAGBase


def calc_price(usage):
    input_price_per_million = 0.15
    output_price_per_million = 0.40

    # Supports OpenAI Responses API and Chat Completions API usage fields
    input_tokens = getattr(
        usage, "input_tokens",
        getattr(usage, "prompt_tokens", 0)
    )
    output_tokens = getattr(
        usage, "output_tokens",
        getattr(usage, "completion_tokens", 0)
    )

    input_cost = input_tokens / 1_000_000 * input_price_per_million
    output_cost = output_tokens / 1_000_000 * output_price_per_million

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def calc_total_price(usages):
    return sum(calc_price(usage)["total_cost"] for usage in usages)


def llm_structured(
    client,
    instructions,
    user_prompt,
    output_type,
    model="glm-5.1",
    max_tokens=1000,
):
    json_instructions = (
        instructions
        + "\n\nReturn only valid JSON matching this schema:\n"
        + output_type.model_json_schema().__str__()
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": json_instructions},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
    )

    raw_content = response.choices[0].message.content

    if not raw_content:
        raise ValueError("The model returned an empty response.")

    parsed_output = output_type.model_validate_json(raw_content)

    return parsed_output, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="glm-5.1",
    max_retries=3,
    max_tokens=1000,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client=client,
                instructions=instructions,
                user_prompt=user_prompt,
                output_type=output_type,
                model=model,
                max_tokens=max_tokens,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {
            "question": 1.0,
            "answer": 2.0,
            "section": 0.1,
        }
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def llm(self, prompt):
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt},
            ],
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.choices[0].message.content

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda _: progress.update())
            futures.append(future)

        for future in futures:
            results.append(future.result())

    return results
