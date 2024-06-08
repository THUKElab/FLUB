import json
import mlcroissant as mlc

# FileObjects and FileSets define the resources of the dataset.
distribution = [
    # gpt-3 is hosted on a GitHub repository:
    mlc.FileObject(
        id="github-repository",
        name="github-repository",
        description="Tsinghua Knowledge Engineering Laboratory (SZ)",
        content_url="https://github.com/THUKElab/FLUB",
        encoding_format="git+https",
        sha256="main",
    ),
    # Within that repository, a FileSet lists all JSONL files:
    mlc.FileSet(
        id="jsonl-files",
        name="jsonl-files",
        description="JSONL files are hosted on the GitHub repository.",
        contained_in=["github-repository"],
        encoding_format="application/jsonl",
        includes="data/*.jsonl",
    ),
]
record_sets = [
    # RecordSets contains records in the dataset.
    mlc.RecordSet(
        id="jsonl",
        name="jsonl",
        # Each record has one or many fields...
        fields=[
            # Fields can be extracted from the FileObjects/FileSets.
            mlc.Field(
                id="jsonl/id",
                name="id",
                description="The id of each data sample.",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    # Extract the field from the column of a FileObject/FileSet:
                    extract=mlc.Extract(column="id"),
                ),
            ),
            mlc.Field(
                id="jsonl/text",
                name="text",
                description="The input cunning text.",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(column="text"),
                ),
            ),
            mlc.Field(
                id="jsonl/is_question",
                name="is_question",
                description="Is the input cunning text a question?",
                data_types=mlc.DataType.BOOL,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(column="is_question"),
                ),
            ),
            mlc.Field(
                id="jsonl/type",
                name="type",
                description="The cunning type of the input text for the Cunning Type Classification task.",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(column="type"),
                ),
            ),
            mlc.Field(
                id="jsonl/explanation",
                name="explanation",
                description="The correct explanation of the input text for the Fallacy Explanation task.",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(column="explanation"),
                ),
            ),
            mlc.Field(
                id="jsonl/options",
                name="options",
                description="The candidate answers for the input text (question).",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(column="options"),
                ),
            ),
            mlc.Field(
                id="jsonl/answer",
                name="answer",
                description="The correct answer for the Answer Selection (Multiple Choice) task.",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(column="answer"),
                ),
            ),
        ],
    )
]

# Metadata contains information about the dataset.
metadata = mlc.Metadata(
    name="FLUB",
    # Descriptions can contain plain text or markdown.
    description=("In this paper, we challenge the reasoning and understanding abilities of LLMs"
                 " by proposing a FaLlacy Understanding Benchmark (FLUB) containing cunning texts"
                 " that are easy for humans to understand but difficult for models to grasp. "
                 "Specifically, the cunning texts that FLUB focuses on mainly consist of the tricky, "
                 "humorous, and misleading texts collected from the real internet environment. "
                 "FLUB has 8 fine-grained types of cunning texts and most of the texts in FLUB "
                 "fall into two types of fallacy, namely, faulty reasoning and word game. "
                 "Moreover, we also manually annotated one correct answer (i.e., the explanation "
                 "of the cunning text) and three confusing wrong answers for each input text in FLUB. "
                 "Based on FLUB and its annotation information, we design three tasks with increasing "
                 "difficulty to test whether the LLMs can understand the fallacy and solve the ``cunning'' texts."
                 " Specifically, (1) Answer Selection: The model is asked to select the correct one from the "
                 "four answers provided by FLUB for each input text. (2) Cunning Type Classification:"
                 " Given a cunning text as input, the model is expected to directly identify its fallacy type defined in our scheme."
                 " (3) Fallacy Explanation: We hope the model sees a cunning text and intelligently generates "
                 "a correct explanation for the fallacy contained in the text, just like humans, without falling into its trap."),
    cite_as=("@article{li2024llms,title={When llms meet cunning texts: "
             "A fallacy understanding benchmark for large language models}, "
             "author={Li, Yinghui and Zhou, Qingyu and Luo, Yuanzhen and Ma, Shirong "
             "and Li, Yangning and Zheng, Hai-Tao and Hu, Xuming and Yu, Philip S}, "
             "journal={arXiv preprint arXiv:2402.11100}, year={2024}}"),
    url="https://github.com/THUKElab/FLUB",
    distribution=distribution,
    record_sets=record_sets,
)

with open("FLUB_croissant_metadata.json", "w") as f:
    content = metadata.to_json()
    content = json.dumps(content, indent=2)
    print(content)
    f.write(content)
    f.write("\n")  # Terminate file with newline