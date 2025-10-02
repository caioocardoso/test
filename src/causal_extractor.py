import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS

class CausalExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.max_length = 3000000
        self.causal_patterns = [
            'cause', 'lead', 'result', 'produce', 'contribute', 'associate', 
            'link', 'affect', 'influence', 'induce', 'trigger', 'promote', 
            'inhibit', 'prevent', 'increase', 'decrease', 'reduce', 'improve'
        ]

    def _clean_entity_text(self, text):
        cleaned = re.sub(r'\[[\d\s,;-]+\]|\(\s*\d+\s*\)', '', text)
        cleaned = re.sub(r'[^\w\s-]', ' ', cleaned)
        cleaned = re.sub(r'\s-\s|\s-\b|\b-\s', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
        
        if not cleaned or len(cleaned) <= 3: return None
        
        stop_words_boundary = set(STOP_WORDS)
        stop_words_boundary.update(['the', 'a', 'an', 'of', 'in', 'for', 'on', 'with', 'as', 'by', 'at', 'to'])
        words = cleaned.split()
        while words and words[0] in stop_words_boundary: words.pop(0)
        while words and words[-1] in stop_words_boundary: words.pop()
        
        if not words: return None
        cleaned = ' '.join(words)
        
        generic_terms = {'study', 'result', 'research', 'analysis', 'data', 'finding', 'increase', 'decrease', 'effect', 'change', 'group', 'level', 'rate', 'author', 'method', 'conclusion', 'introduction', 'expression', 'risk', 'example'}
        if cleaned in generic_terms: return None
        
        return cleaned if len(cleaned) >= 4 else None

    def _is_meaningful_relationship(self, cause, effect):
        if not cause or not effect or cause == effect: return False
        generic_terms = {'study', 'result', 'research', 'analysis', 'data', 'finding'}
        if cause in generic_terms or effect in generic_terms: return False
        
        cause_words = set(cause.split())
        effect_words = set(effect.split())
        if not cause_words or not effect_words: return False
        
        overlap = len(cause_words & effect_words) / min(len(cause_words), len(effect_words))
        return overlap < 0.7

    def extract_causal_relationships(self, doc):
        relationships = []
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in self.causal_patterns:
                    causes, effects = [], []
                    for child in token.children:
                        if child.dep_ in ('nsubj', 'nsubjpass'):
                            causes.extend([child] + list(child.conjuncts))
                    for child in token.children:
                        if child.dep_ in ('dobj', 'attr'):
                            effects.extend([child] + list(child.conjuncts))
                    if not effects:
                        for prep in (c for c in token.children if c.dep_ == 'prep'):
                            for pobj in (p for p in prep.children if p.dep_ == 'pobj'):
                                effects.extend([pobj] + list(pobj.conjuncts))
                    
                    if causes and effects:
                        for cause_token in causes:
                            for effect_token in effects:
                                cause_text = next((c.text for c in sent.noun_chunks if c.start <= cause_token.i < c.end), cause_token.text)
                                effect_text = next((c.text for c in sent.noun_chunks if c.start <= effect_token.i < c.end), effect_token.text)
                                cause_clean = self._clean_entity_text(cause_text)
                                effect_clean = self._clean_entity_text(effect_text)
                                if self._is_meaningful_relationship(cause_clean, effect_clean):
                                    relationships.append((cause_clean, effect_clean))
        return relationships

    def extract_from_text(self, text):
        doc = self.nlp(text)
        relationships = self.extract_causal_relationships(doc)
        
        entities = set()
        for cause, effect in relationships:
            entities.add(cause)
            entities.add(effect)
            
        return relationships, entities