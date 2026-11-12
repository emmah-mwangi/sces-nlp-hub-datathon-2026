# Supported task choices, kit v2.0
## A. Intent classification (default)
10 MASSIVE-aligned labels in taxonomy.csv. Minimum 250 accepted contribution rows: per label 15 train, 5 dev, 5 test. This is a small-data prototype target. Optional stretch 700 = (50 train +10 dev +10 test) x10, not a new compulsory threshold.
Evaluate accuracy, macro-F1, per-label F1 and a majority baseline. Show request -> label -> simulated action.
## B. Sentiment analysis
Three labels: positive, negative, neutral. Minimum 300 accepted contribution rows: per label 60 train,20 dev,20 test. Scope: sentence-level opinion about one declared domain such as campus services. Keep ambiguous mixed opinion or sarcasm out of this starter scope and log exclusions.
Evaluate accuracy, macro-F1, class counts, confusion matrix and majority baseline. Show text -> opinion label. Do not treat sentiment as intent or emotion.
Sources: AfriSenti where supported, or permitted new human examples/annotations. Publisher: https://github.com/afrisenti-semeval/afrisent-semeval-2023 . Original publisher README states CC BY 4.0; pin the exact source/version.
## C. Limited semantic parsing (advanced)
Use the same 10 action labels plus surface slots drawn from location,date,time,item,term. Minimum250 accepted rows with at least100 nonempty-slot rows. Per action15 train,5 dev,5 test; include both empty and nonempty slots in test and reserve new slot values for an explicit challenge subset.
Schema: slots_json is a JSON list of {type,value,start,end}; offsets use Python Unicode character indices, start-inclusive/end-exclusive; text[start:end] must equal value. [] means no slot.
Example: text='Hali ya hewa Nairobi kesho?'
intent=weather_query
slots=[{'type':'location','value':'Nairobi','start':13,'end':20},{'type':'date','value':'kesho','start':21,'end':26}]
Verify the offsets yourself, do not copy untested positions.
Model: intent classifier plus training-only memorised surface-value matcher. This is a limited baseline, not an open-ended semantic parser; report its unseen-value failures.
Evaluate intent macro-F1, slot type+span micro-F1, full-frame exact match and valid JSON rate. The starter serialises valid JSON by construction; this does not establish correct meaning. No nested logic, text-to-SQL, code execution, real actions or date/time conversion.
## Other tasks
NER (MasakhaNER 2.0) or topic classification (KINNEWS) can be approved only with a written schema, target, evaluator and mentor. They are not implemented by these three-track scripts. Do not feed NER token labels into the sentiment or intent validator.
Translation, audio and open-ended SQL/parsing are outside the supported starter scope.
## Common
At least250/300 contribution rows means new annotations, permitted human localisation or human-authored examples, not unchanged downloads. Use group_id for shared sources/paraphrases. Review all test rows independently. Keep exact sources, attribution, exclusions, AI use and changes.
Do not force Common Voice text into opinion or intent labels. It has no automatic labels for these tasks.

