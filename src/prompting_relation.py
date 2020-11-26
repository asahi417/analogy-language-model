""" Prompting function """
import json

TEMPLATES = {
    'is-to-what': "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>",
    'is-to-as': "<subj-a> is to <obj-a> as <subj-b> is to <obj-b>",
    'rel-same': 'The relation between <subj-a> and <obj-a> is the same as the relation between <subj-b> and <obj-b>',
    'what-is-to': 'what <subj-a> is to <obj-a>, <subj-b> is to <obj-b>'
}

__all__ = ('prompting_relation', 'get_prompt_dataset', 'TEMPLATES')


def prompting_relation(subject_stem,
                       object_stem,
                       subject_analogy,
                       object_analogy,
                       template: str = None,
                       template_type: str = 'is-to-what'):
    """ to convert a SAT style analogy set into a natural sentence with a template """
    if template is None:
        template = TEMPLATES[template_type]
    assert "<subj-a>" in template and "<subj-b>" in template and "<obj-a>" in template and "<obj-b>" in template
    template = template.replace("<subj-a>", subject_stem).replace("<obj-a>", object_stem).\
        replace("<subj-b>", subject_analogy).replace("<obj-b>", object_analogy)
    return template


def get_prompt_dataset(path_to_data: str, template_type: str = 'is-to-what'):
    """ get prompted SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""

    def format_entry(dictionary):
        prompts = [
            prompting_relation(
                subject_stem=dictionary['stem'][0],
                object_stem=dictionary['stem'][1],
                subject_analogy=c[0],
                object_analogy=c[1],
                template_type=template_type
            ) for c in dictionary['choice']]

        return dictionary['answer'], prompts, dictionary['stem'], dictionary['choice']

    with open(path_to_data, 'r') as f:
        data = [format_entry(json.loads(i)) for i in f.read().split('\n') if len(i) > 0]

    list_answer = list(list(zip(*data))[0])
    list_nested_sentence = list(list(zip(*data))[1])
    list_stem = list(list(zip(*data))[2])
    list_choice = list(list(zip(*data))[3])
    return list_answer, list_nested_sentence, list_stem, list_choice
