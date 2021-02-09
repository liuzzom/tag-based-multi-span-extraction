from src.data.dataset_readers.drop.drop_reader import DropReader
from src.data.tokenizers.huggingface_transformers_tokenizer import HuggingfaceTransformersTokenizer
from src.data.dataset_readers.answer_field_generators.tagged_answer_generator import TaggedAnswerGenerator
from src.data.dataset_readers.answer_field_generators.arithmetic_answer_generator import ArithmeticAnswerGenerator
from src.data.dataset_readers.answer_field_generators.count_answer_generator import CountAnswerGenerator
from src.data.dataset_readers.answer_field_generators.span_answer_generator import SpanAnswerGenerator

tokenizer = HuggingfaceTransformersTokenizer("roberta-large")

answer_field_generators = {
    "tagged_answer": TaggedAnswerGenerator(ignore_question=False),
    "arithmetic_answer": ArithmeticAnswerGenerator(special_numbers=[100, 1]),
    "count_answer": CountAnswerGenerator(),
    "passage_span_answer": SpanAnswerGenerator(text_type="passage"),
    "question_span_answer": SpanAnswerGenerator(text_type="question"),
}

answer_generator_names_per_type = {
    "multiple_span": ["passage_span_answer", "tagged_answer"],
    "single_span": ["passage_span_answer", "question_span_answer", "tagged_answer"],
    "number": ["arithmetic_answer", "count_answer", "passage_span_answer", "question_span_answer", "tagged_answer"],
    "date": ["arithmetic_answer", "passage_span_answer", "question_span_answer", "tagged_answer"]
}

dropReader = DropReader(
    tokenizer=tokenizer,
    answer_field_generators=answer_field_generators,
    answer_generator_names_per_type=answer_generator_names_per_type,
    old_reader_behavior=True)

dropReader.read('drop_data/drop_single_sample.json')
