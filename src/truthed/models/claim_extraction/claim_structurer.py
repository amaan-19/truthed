class ClaimStructurer:
    def structure_claim(self, claim_text: str) -> StructuredClaim:
        # Use spaCy to extract entities and dependencies
        # Return subject, predicate, object, temporal context
        # Mark verifiability (can this claim be fact-checked?)
