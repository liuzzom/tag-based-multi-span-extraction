from allennlp.data import Vocabulary
from allennlp.modules import FeedForward

from src.data.dataset_readers.drop.drop_reader import DropReader
from src.data.tokenizers.huggingface_transformers_tokenizer import HuggingfaceTransformersTokenizer
from src.data.dataset_readers.answer_field_generators.tagged_answer_generator import TaggedAnswerGenerator
from src.data.dataset_readers.answer_field_generators.arithmetic_answer_generator import ArithmeticAnswerGenerator
from src.data.dataset_readers.answer_field_generators.count_answer_generator import CountAnswerGenerator
from src.data.dataset_readers.answer_field_generators.span_answer_generator import SpanAnswerGenerator
from src.models.multi_head_model import MultiHeadModel
from src.modules.heads.arithmetic_head import ArithmeticHead
from src.modules.heads.count_head import CountHead
from src.modules.heads.multi_span_head import MultiSpanHead
from src.modules.heads.passage_span_head import PassageSpanHead
from src.modules.heads.question_span_head import QuestionSpanHead

""" Creating Dataset Reader """
tokenizer = HuggingfaceTransformersTokenizer(pretrained_model="roberta-large")

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

dropReader = DropReader(
    tokenizer=tokenizer,
    answer_field_generators=answer_field_generators,
    answer_generator_names_per_type=answer_generator_names_per_type,
    is_training=True,
    old_reader_behavior=True,
    pickle=pickle)

instances = dropReader.read('drop_data/drop_single_sample.json')

""" Build Vocabulary """
print("Building Vocab...")
vocab = Vocabulary.from_instances(instances)
print("...Created")

""" Create Model """
print("Building Head Predictor...")
head_predictor = FeedForward(
    activations=["relu", "linear"],
    dropout=[0.1, 0],
    hidden_dims=[1024, 5],
    input_dim=1024,
    num_layers=2
)
print("...Created")

print("Building Heads...")
heads = {
    "arithmetic": ArithmeticHead(
        output_layer=FeedForward(
            activations=["relu", "linear"],
            dropout=[0.1, 0],
            hidden_dims=[1024, 3],
            input_dim=2048,
            num_layers=2
        ),
        special_embedding_dim=1024,
        special_numbers=[100, 1],
        training_style="soft_em"
    ),
    "count": CountHead(
        max_count=10,
        output_layer=FeedForward(
            activations=["relu", "linear"],
            dropout=[0.1, 0],
            hidden_dims=[1024, 11],
            input_dim=1024,
            num_layers=2
        )
    ),
    "multi_span": MultiSpanHead(
        decoding_style="at_least_one",
        ignore_question=False,
        labels={"I": 1, "O": 0},
        output_layer=FeedForward(
            activations=["relu", "linear"],
            dropout=[0.1, 0],
            hidden_dims=[1024, 11],
            input_dim=1024,
            num_layers=2
        ),
        prediction_method="viterbi",
        training_style="soft_em"
    ),
    "passage_span": PassageSpanHead(
        end_output_layer=FeedForward(
            activations=["linear"],
            hidden_dims=1,
            input_dim=1024,
            num_layers=1
        ),
        start_output_layer=FeedForward(
            activations=["linear"],
            hidden_dims=1,
            input_dim=1024,
            num_layers=1
        ),
        training_style="soft_em"
    ),
    "question_span": QuestionSpanHead(
        end_output_layer=FeedForward(
            activations=["relu", "linear"],
            dropout=[0.1, 0],
            hidden_dims=[1024, 1],
            input_dim=2048,
            num_layers=2
        ),
        start_output_layer=FeedForward(
            activations=["relu", "linear"],
            dropout=[0.1, 0],
            hidden_dims=[1024, 1],
            input_dim=2048,
            num_layers=2
        ),
        training_style="soft_em"
    )
}
print("...Created")

print("Building Passage Summary Vector Module...")
passage_summary_vector_module = FeedForward(
    activations=["linear"],
    hidden_dims=1,
    input_dim=1024,
    num_layers=1
)
print("...Created")

print("Building Question Summary Vector Module...")
question_summary_vector_module = FeedForward(
    activations=["linear"],
    hidden_dims=1,
    input_dim=1024,
    num_layers=1
)
print("...Created")

print("Building Model...")
model = MultiHeadModel(
    dataset_name="drop",
    head_predictor=head_predictor,
    heads=heads,
    passage_summary_vector_module=passage_summary_vector_module,
    pretrained_model="roberta-large",
    question_summary_vector_module=question_summary_vector_module,
    vocab=vocab
)
print("...Created")

test_instance = instances[0]
test_fields = test_instance.fields

model.forward(
    question_passage_tokens=test_fields['question_passage_tokens'].as_tensor(),
    question_passage_token_type_ids=test_fields['question_passage_token_type_ids'].as_tensor(),
    question_passage_special_tokens_mask=test_fields['question_passage_special_tokens_mask'].as_tensor(),
    question_passage_pad_mask=test_fields['question_passage_pad_mask'].as_tensor(),
    first_wordpiece_mask=test_fields['first_wordpiece_mask'].as_tensor(),
    number_indices=test_fields['number_indices'].as_tensor(),
    answer_as_expressions=test_fields['answer_as_expressions'].as_tensor(),
    answer_as_expressions_extra=test_fields['answer_as_expressions_extra'].as_tensor(),
    answer_as_counts=test_fields['answer_as_counts'].as_tensor(),
    answer_as_passage_spans=test_fields['answer_as_passage_spans'].as_tensor(),
    answer_as_question_spans=test_fields['answer_as_question_spans'].as_tensor(),
    wordpiece_indices=test_fields['wordpiece_indices'].as_tensor(),
    answer_as_text_to_disjoint_bios=test_fields['answer_as_text_to_disjoint_bios'].as_tensor(),
    answer_as_list_of_bios=test_fields['answer_as_list_of_bios'].as_tensor(),
    span_bio_labels=test_fields['span_bio_labels'].as_tensor(),
    is_bio_mask=test_fields['is_bio_mask'].as_tensor(),
    metadata=test_fields['metadata'].as_tensor()
)
