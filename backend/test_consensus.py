from app.services.consensus import get_consensus

result = get_consensus("What is 2+2? Answer in one short sentence.")

print("Final Answer:", result.answer)
print("Models used:", result.models_used)
print("Ranking (weights applied):", result.ranking)
print("Agreement score:", result.agreement_score)
print("Contradictions:", result.contradictions)
print("Confidence:", result.confidence)
print("Synthesis notes:", result.synthesis_notes)