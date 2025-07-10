class ClaimExtractionPipeline:
    def extract_claims(self, url: str) -> List[Claim]:
        # 1. Scrape article content
        # 2. Clean and preprocess text  ✅ (already done)
        # 3. Extract sentences ✅ (already done)
        # 4. Run BERT classifier on each sentence
        # 5. Structure the identified claims
        # 6. Return list of Claim objects
