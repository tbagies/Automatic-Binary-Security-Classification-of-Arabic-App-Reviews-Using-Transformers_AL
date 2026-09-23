---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:300
- loss:BatchHardTripletLoss
base_model: UBC-NLP/MARBERTv2
widget:
- source_sentence: الحذر ثم الحذر ثم الحذر ان تطبيق واتس هو تطبيق نصب واحتيال وشبكة
    تجسسية والله ثم والله ثم والله لقد تم الاسيلاء على حسابي من قبل شركة واتساب وتم
    حظر رقمي من غير اي سبب ولا استطيع الوصول الى حسابي الان وحاولت مرارا وتكرارا دون
    فايدة وانا خسرت الكثير بسبب هذا الحظر والله على ما اقول شهيد ولهذا احذركم ان تثقو
    بهذا التطبيق وانهم كاذبون عندما يقولون انها تطبيق مشفر
  sentences:
  - مش راضي يثبت كل لما إثبته يعمل إلغاء تاني ثبته اكتر من مره وميثبتش ليه
  - ليشش غير عربي
  - تجسس ميزة ان مشتركين البلس يقدرون يشوفون الرسائل المحذوفة سيئة جدا واتمنى حذفها
    لانها تنتهك الخصوصية
- source_sentence: برنامج تجسسي واحتيال ارجو الانتباه والابلاغ عن التطبيق
  sentences:
  - تنسو أن أصحاب البرامج دى هدفهم الحقيقى هو دس أنوفهم فى خصوصية المستخدمين والعرب
    على بوجه خاصبمعنى أوضح التجسس على المستخدمين العرب والمسلمين بصفه عامه
  - برنامج تجسسي وينتهك الخصوصية الفردية
  - تطبيق للتجسس وانتهاك الخصوصية والتسول والمتاجرة بالأعضاء البشرية وسرقة الاموال
    بحجة التبرع
- source_sentence: برنامج فاشل ويتجسس على مستخدميه ولا يراعي خصوصية المستخدمين نهائيا
    ، اسوء برامج جوجل بلاي ولو فيه أقل من نجمة كنت أعطيته
  sentences:
  - كما اعتقد ان هذا التطبيق مستمر رغم كل الانتقادات من اغلب المستخدمين إلا ان القائمين
    علا العمل بهذا التطبيق متجاهلين تماما مانقوله وكما اتوقع انه يخدم المطورين بشيئ
    ما ربما معلومات سريه اومارقبه اشخاص او تجسس وتسريب صور خاصه وفيديو وإتصال صوت
    او صورة خاص لذالك انصح كل الاخوة وغيرهم من المستخدمين ان لايضيعو وقتهم وجهدهم
    ومالهم في عمل فاشل مثل ماسنجر فاشل كان بدايه إنطلاق فيسبوك بميزة الرسائل ولا اروع
    والان اصبح لديه ميزه شبه فاشله وشبه ناجحه فيه ميزة الاتصال مقبوله وميزة الرسائل
    صعبة بعض الشيئ وحجم مبالغ فيه
  - السعودية توجد مشكلة عند الدخول للتطبيق فهو لا يستجيب إلا بعد عدة محاولات وهذا
    شي محبط و يطفشش
  - مايفتح ليش التطبيق مايفتح حدثته بس ماينفتح
- source_sentence: تطبيق فاشل ما يشتغل زين على جوالي و عندي ايفون والله يشتغل عليه
    زي الفلل
  sentences:
  - فيسبوك ده غير امن وممكن يكون موقع اسرائيلي للتجسس وغير كده نيات الفيسبوك وحشه
    وانصح اي حد بيفكر انه يسجل يفكر الف مره قبل مايعمل كده واللهم بلغت اللهم فاشهد
  - السلام عليكم
  - برنامج به اعطال من الناحية الفنية للبرنامج به عطل يغلق فجأة بمجرد الضغط على إبلاغ
    عن حادث و للاسف معاملة المحققين غاية في السوء و استهزاء بالناس و قد حدث موقفين
    بنفس الكيفية
- source_sentence: تطبيق غير آمن يتجسس علي المستخدمين مسح تحويل لتليجرام
  sentences:
  - برنامج خبيث وسئ تحدثت مع ال عن مشاريع وفجأة وبدون انذار توقف وأرسل لى رسالة إنى
    ارسلت رساله خالفت القوانين من اين له أن يعرف انى ارسلت رسالة مخالفة للقانون والرسائل
    كما يزعمون مشفرة إذا هناك من يتجسس على رسائل الواتس أو ربما لا تعجبهم منشوراتى
    لأنهم عنصريون
  - يقيد حرية الأراء حسب معتقدات مالكه يتجسس عن الخصوصية للفرد
  - سيئ جدا تقنيا يعلق ٢٤ ساعة يحتاج نت قوي وهذا يدل على ركاكة التطبيق تتبعي تجسسي
    منتهك للخصوصية والحرية الفردية لذلك تقييمه سيئ ولابد ان يلغى فورا وشكرا
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on UBC-NLP/MARBERTv2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [UBC-NLP/MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [UBC-NLP/MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2) <!-- at revision fe88db9db8ccdb0c4e1627495f405c44a5f89066 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'PeftModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'تطبيق غير آمن يتجسس علي المستخدمين مسح تحويل لتليجرام',
    'يقيد حرية الأراء حسب معتقدات مالكه يتجسس عن الخصوصية للفرد',
    'سيئ جدا تقنيا يعلق ٢٤ ساعة يحتاج نت قوي وهذا يدل على ركاكة التطبيق تتبعي تجسسي منتهك للخصوصية والحرية الفردية لذلك تقييمه سيئ ولابد ان يلغى فورا وشكرا',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9860, 0.9875],
#         [0.9860, 1.0000, 0.9907],
#         [0.9875, 0.9907, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 300 training samples
* Columns: <code>sentence_0</code> and <code>label</code>
* Approximate statistics based on the first 300 samples:
  |         | sentence_0                                                                         | label                                           |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------|
  | type    | string                                                                             | int                                             |
  | details | <ul><li>min: 3 tokens</li><li>mean: 32.58 tokens</li><li>max: 246 tokens</li></ul> | <ul><li>0: ~50.00%</li><li>1: ~50.00%</li></ul> |
* Samples:
  | sentence_0                                                                                                                                         | label          |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------|:---------------|
  | <code>أصبح هذا التحديث يتجسس على الخصوصيه أعطو نجمه واحدا فقط</code>                                                                               | <code>1</code> |
  | <code>جيد للدراسة لاكن يمك اختراق الحساب و فتح الكامرة و المايك عند الإجتماع و التجسس ايضا و يمكن ان يخرجوك من الاجتماع ارجو حل هذه المشكلة</code> | <code>1</code> |
  | <code>مقدر اسجل صوتي ولا اقدر اسوي شي اتمنى تعدلونه</code>                                                                                         | <code>0</code> |
* Loss: [<code>BatchHardTripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#batchhardtripletloss)

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.9.0+cu128
- Accelerate: 1.11.0
- Datasets: 4.3.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### BatchHardTripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->