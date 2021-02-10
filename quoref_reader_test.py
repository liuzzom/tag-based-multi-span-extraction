from src.data.dataset_readers.answer_field_generators.arithmetic_answer_generator import ArithmeticAnswerGenerator
from src.data.dataset_readers.answer_field_generators.count_answer_generator import CountAnswerGenerator
from src.data.dataset_readers.answer_field_generators.span_answer_generator import SpanAnswerGenerator
from src.data.dataset_readers.answer_field_generators.tagged_answer_generator import TaggedAnswerGenerator
from src.data.dataset_readers.quoref.quoref_reader import QuorefReader
from src.data.tokenizers.huggingface_transformers_tokenizer import HuggingfaceTransformersTokenizer

answer_field_generators = {
    "arithmetic_answer": ArithmeticAnswerGenerator(special_numbers=[100, 1]),
    "count_answer": CountAnswerGenerator(),
    "passage_span_answer": SpanAnswerGenerator(text_type="passage"),
    "question_span_answer": SpanAnswerGenerator(text_type="question"),
    "tagged_answer": TaggedAnswerGenerator(ignore_question=False, labels={"I": 1, "O": 0})
}

answer_generator_names_per_type = {
    "date": ["arithmetic_answer", "passage_span_answer", "question_span_answer", "tagged_answer"],
    "multiple_span": ["tagged_answer"],
    "single_span": ["passage_span_answer", "question_span_answer", "tagged_answer"],
    "number": ["arithmetic_answer", "count_answer", "passage_span_answer", "question_span_answer", "tagged_answer"],
}

pickle = {"action": None, "file_name": "all_heads_IO_roberta-large", "path": "../pickle/drop"}

tokenizer = HuggingfaceTransformersTokenizer(pretrained_model="roberta-large")

quorefReader = QuorefReader(
    answer_field_generators=answer_field_generators,
    answer_generator_names_per_type=answer_generator_names_per_type,
    is_training=True,
    old_reader_behavior=True,
    pickle=pickle,
    tokenizer=tokenizer)

quorefReader.read('quoref_data/quoref_dataset_dev.json')