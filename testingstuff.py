from claim_pipeline import get_claim_probabilities
import spacy

nlp = spacy.load("en_core_web_sm")

article_text = """"Thirty-three centuries had passed since human feet last trod the floor on which we stood, and yet the signs of recent life were around us."

Ninety years on, Howard Carter's clipped English voice sounds itself almost like an ancient relic. When he spoke on BBC radio in 1936, 14 years had gone by since he first uncovered the treasure-rich tomb of Tutankhamun. The almost miraculous discovery by Carter and his expert team of the boy king's intact tomb had made him world-famous and sparked a craze for all things ancient Egyptian. 

Speaking on a programme reviewing the events of 1924, he conjured the uncanny sensation he felt on 12 February that year when they finally reached Tutankhamun's sarcophagus, the stone coffin where the pharaoh had lain undisturbed for millennia. When he notes details such as "a half‑filled bowl of mortar, a blackened lamp, the chips of wood left on the floor by a careless carpenter", his sense of wonder comes through as alive as ever.Despite going on to make one of the world's greatest archaeological discoveries, Carter had left school at 15 and had no formal training. With his talent for drawing, a local aristocratic family who lived near his rural Norfolk home took the solitary teenager under their wing. The Amhersts' Didlington Hall had the greatest private collection of Egyptian objects in Britain, and he became fascinated by their stories. At 17, his artistic skill secured him work in Egypt as a draughtsman and tracer. He arrived during a boom for archaeology, much of it funded by wealthy amateurs and British aristocrats. For more than two decades he trained on the job. 

The staggering discovery of Tutankhamun's tomb owed much to good fortune. Carter had been toiling away with little success for years in the Valley of the Kings, an area just west of the Nile used by the ancient Egyptians as the main burial ground for the pharaohs. The tomb entrance had long been concealed by layers of ancient debris, keeping it beyond the reach of both grave robbers and archaeologists. """

doc = nlp(article_text)
sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

print(f"Total sentences found by spaCy: {len(sentences)}\n")

probs = get_claim_probabilities(sentences)

for i, (sent, p) in enumerate(zip(sentences, probs), 1):
    print(f"[{i}] P(claim)={p:.3f}")
    print(f"    {sent[:150]}")
    print()