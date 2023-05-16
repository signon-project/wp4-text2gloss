"""some lists that are used: """
from helpers import read_file
import os

pers1_personal = ('ik', 'ikke', "'k")                           # I
pers1_demonstrative = ('mijn', 'mij', 'me', "m'n")              # me, mine
pers1 = pers1_personal + pers1_demonstrative
pers2_personal = ('jij', 'je', 'jou', 'u', 'gij', 'ge')         # you
pers2_demonstrative = ('je', 'jouw', 'uw')                      # you, your
pers2 = pers2_personal + pers2_demonstrative
pers4_personal = ('wij', 'we')                                  # we
pers4_demonstrative = ('ons', 'onze')                           # us, our
pers4 = pers4_personal + pers4_demonstrative
pers5 = ('jullie',)                                              # you
pers3_6_personal = ('hij', 'zij', 'ze', 'het', 'hen', 'hun')    # he, she, it
pers3_6_demonstrative = ('zijn', 'hem', 'haar', 'hen', 'hun', 'die', 'dat', 'deze', 'dit', "z'n")
                                                                # him, his, her, it, its
pers3_6 = pers3_6_personal + pers3_6_demonstrative
demonstrative = pers1_demonstrative + pers2_demonstrative + pers4_demonstrative + pers5 + pers3_6_demonstrative
####################################################################################################################
locative_verbs = ('liggen', 'staan', 'gaan', 'zitten')                    # lie, stand, go
# 'zitten' can be a locative verb that needs to be deleted, eg: "er zitten vlekken op het raam" >< "de kat zit"

locative_preps = ('aan', 'eraan',
                  'achter',  'erachter', 'achterom',
                  'af', 'eraf',  'vanaf', 'ervanaf',
                  'bij', 'erbij', 'nabij'
                  'binnen', 'binnenin', 'erbinnen',
                  'boven', 'bovenop', 'erboven', 'erbovenop', 'bovenaan',
                  'buiten', 'erbuiten',
                  'door', 'erdoor', 'doorheen', 'erdoorheen',
                  'in', 'erin',
                  'naast', 'ernaast',
                  'langs', 'erlangs',
                  'naar', 'ernaar', 'naartoe', 'ernaartoe',
                  'om','erom',
                  'onder', 'eronder',
                  'op', 'erop',
                  'over', 'erover', 'achterover'
                  'richting',
                  'rond',  'errond', 'rondom',
                  'tegen', 'ertegen', 'tegenover', 'tegenaan', 'ertegenenaan'
                  'toe',  'ertoe',
                  'tussen', 'ertussen', 'tussenin', 'ertussenin',
                  'uit', 'eruit', 'vanuit',
                  'van', 'ervan', 'ervanaf', 'vanaf'
                  'voor', 'ervoor', 'voorom')

# conjugations because the lemma in spacy is not always the infinitive for linking verbs
be_conjugation = ('zijn', 'ben', 'bent', 'is', 'waren', 'geweest', 'was')               # to be
worden_conjugation = ('worden', 'word', 'wordt', 'werd', 'werden', 'geworden')          # worden = to be in passive
blijven_conjugation = ('blijven', 'blijf', 'blijft', 'bleef', 'gebleven')               # keep on
blijken_conjugation = ('blijken', 'blijk', 'blijkt', 'bleken', 'gebleken', 'bleek')     # appear to be
lijken_conjugation = ('lijken', 'lijk', 'lijkt', 'leek')                                # seem to
have_conjugation = ('hebben', 'heb', 'heeft', 'hebt', 'hadden', 'gehad')                # to have
will_conjugation = ('zullen', 'zal', 'zult', 'zou', 'zouden')                           # will


not_signed_verbs = be_conjugation + worden_conjugation + blijven_conjugation + blijken_conjugation \
                   + lijken_conjugation + have_conjugation + will_conjugation
# no linking verbs in VGT
# 'duren' not signed, eg: 'het kan lang duren', 'het duurt 2 uur'

signed_if_only_verb = ('gaan', 'blijven', 'zitten')  # go, stay, sit
schijnen_literal = ('zon', 'licht')  # to shine (>< seem to)
####################################################################################################################
verb_with_congruence = ('geven',) # ([)'geven', 'verwittigen', 'achtervolgen', 'uitnodigen', 'krijgen', 'komen']
# this is not complete
# some parameter of the sign changes based on subj/obj
####################################################################################################################
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

with open(os.path.join(__location__, 'scheidbare_ww.txt'), 'r') as sep_file:
    separable_verbs = sep_file.read().split('\n')

with open(os.path.join(__location__, 'scheidbare_ww_scheidbaar_deel.txt'), 'r') as sep_file:
    seperated_part_of_verb = sep_file.read().split('\n')
####################################################################################################################
time_unit = ('seconde', 'minuut', 'kwartier', 'uur', 'dag', 'week', 'maand', 'jaar', 'eeuw', 'millenium', 'tijd')
time_point = ('eergisteren', 'gisteren', 'gister', 'vandaag', 'nu', 'morgen', 'overmorgen', 'vanavond',
              'vanochtend', 'vanmiddag', 'vannacht', 'vroeger', 'verleden', 'toekomst', 'vroeg', 'laat')
season = ('lente', 'zomer', 'herfst', 'winter', 'voorjaar', 'najaar')
day_of_week = ('maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag', 'weekend')
month = ('januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli', 'augustus', 'september', 'oktober',
         'november', 'december')
moment_of_day = ('ochtend', 'morgen', 'voormiddag', 'middag', 'namiddag', 'vooravond', 'avond', 'nacht', 'dag',
                 'zonsopgang', 'zonsondergang')
repeat_time = ('dagelijks', 'wekelijks', 'maandelijks', 'jaarlijks')  # every ...
temporal_expressions = time_unit + time_point + season + day_of_week + month + moment_of_day \
                       + repeat_time
preposition_with_time = ('rond', 'voor', 'na', 'achter', 'van', 'richting', 'tegen', 'om', 'in')
# temporal expressions always occur at the beginning of the VGT sentence
####################################################################################################################
question_words_alone = ('wie', 'wat', 'waar', 'wanneer', 'hoe', 'wiens', 'wier', 'waarom', 'waartoe', 'waarheen',
                        'waaraan', 'waarin', 'waarop', 'waarmee', 'waarvoor', 'waardoor', 'waarachter', 'waaronder',
                        'waaraf', 'hoeveel')
question_words_with_noun = ('welke', 'welk', 'hoeveel')  # which ..., how many ...
question_words = question_words_alone + question_words_with_noun
####################################################################################################################
words_with_plural_sign = ('friet', 'kind', 'ouder')
# this is not complete
####################################################################################################################
may_conjugation = ('mogen', 'mag', 'mocht', 'mochten', 'gemogen')
can_conjugation = ('kunnen', 'kan', 'kun', 'kunt', 'kon', 'konden', 'gekund')
must_conjugation = ('moeten', 'moet', 'moest', 'moesten', 'gemoeten')
may_can = may_conjugation + can_conjugation
separate_negation_sign = ('weten',) + can_conjugation + may_conjugation
####################################################################################################################
conjunctions_then_after = ('nadat', 'zodra')
conjunctions_then_before = ('voordat', 'daarna', 'vervolgens', 'hierna', 'dan', 'nadien')
conjunctions_because = ('aangezien', 'doordat', 'want', 'omdat', 'doordat', 'vermits', 'zodat', 'namelijk', 'immers')
conjunctions_because_switch = ('daarom',)
conjunctions_if = ('als', 'indien')  # 'op voorwaarde dat', 'wanneer'
conjunctions_but = ('maar', 'echter')
not_used_conjunctions = ('toen', 'dat')  # context make them clear
conjunctions_while = ('terwijl',)
####################################################################################################################
adverbs_at_front = ('dan', 'daarna', 'dus', 'vervolgens', 'daarom')
conj_at_front = ('dan', 'omdat', 'als', 'maar')
adverbs_after_temporal_clause = ('misschien', 'ook', 'nog', 'altijd', 'meestal')
####################################################################################################################
one_options = ('1', 'een', 'eén', 'één')
small_units = ("nul", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen", "tien", "elf", "twaalf",
               "dertien", "veertien") + one_options
bigger_units = ("vijftien", "zestien", "zeventien", "achttien", "negentien", "twintig", "dertig", "veertig", "vijftig",
                "zestig", "zeventig", "tachtig", "negentig")
big_units = ("honderd", "duizend", "miljoen", "miljard")
big_units_numbers = ("100", "1000", "1000000", "1000000000")
double_numbers = ('tweeëntwintig', 'tweeentwintig', 'drieëndertig', 'drieendertig', 'vierenveertig', 'vijfenvijftig',
                  'zesenzestig', 'zevenenzeventig', 'achtentachtig', 'negenennegentig')
double_signs = ('twee^twee', 'twee^twee', 'drie^drie', 'drie^drie', 'vier^vier', 'vijf^vijf', 'zes^zes', 'zeven^zeven',
                'acht^acht', 'negen^negen')
ordinal_with_sign = ('eerst', 'eerste', 'tweede', 'derde', 'vierde')

# output of prep numbers:
number_in_numbers = tuple(range(998))
number_in_words = ('nul', 'één', 'twee', 'drie', 'vier', 'vijf', 'zes', 'zeven', 'acht', 'negen', 'tien', 'elf',
                   'twaalf', 'dertien', 'veertien', 'vijftien', 'zestien', 'zeventien', 'achttien', 'negentien',
                   'twintig', 'eenentwintig', 'tweeëntwintig', 'drieëntwintig', 'vierentwintig', 'vijfentwintig',
                   'zesentwintig', 'zevenentwintig', 'achtentwintig', 'negenentwintig', 'dertig', 'eenendertig',
                   'tweeëndertig', 'drieëndertig', 'vierendertig', 'vijfendertig', 'zesendertig', 'zevenendertig',
                   'achtendertig', 'negenendertig', 'veertig', 'eenenveertig', 'tweeënveertig', 'drieënveertig',
                   'vierenveertig', 'vijfenveertig', 'zesenveertig', 'zevenenveertig', 'achtenveertig',
                   'negenenveertig', 'vijftig', 'eenenvijftig', 'tweeënvijftig', 'drieënvijftig', 'vierenvijftig',
                   'vijfenvijftig', 'zesenvijftig', 'zevenenvijftig', 'achtenvijftig', 'negenenvijftig', 'zestig',
                   'eenenzestig', 'tweeënzestig', 'drieënzestig', 'vierenzestig', 'vijfenzestig', 'zesenzestig',
                   'zevenenzestig', 'achtenzestig', 'negenenzestig', 'zeventig', 'eenenzeventig', 'tweeënzeventig',
                   'drieënzeventig', 'vierenzeventig', 'vijfenzeventig', 'zesenzeventig', 'zevenenzeventig',
                   'achtenzeventig', 'negenenzeventig', 'tachtig', 'eenentachtig', 'tweeëntachtig', 'drieëntachtig',
                   'vierentachtig', 'vijfentachtig', 'zesentachtig', 'zevenentachtig', 'achtentachtig',
                   'negenentachtig', 'negentig', 'eenennegentig', 'tweeënnegentig', 'drieënnegentig', 'vierennegentig',
                   'vijfennegentig', 'zesennegentig', 'zevenennegentig', 'achtennegentig', 'negenennegentig',
                   'honderd', 'honderdeen', 'honderdtwee', 'honderddrie', 'honderdvier', 'honderdvijf', 'honderdzes',
                   'honderdzeven', 'honderdacht', 'honderdnegen', 'honderdtien', 'honderdelf', 'honderdtwaalf',
                   'honderddertien', 'honderdveertien', 'honderdvijftien', 'honderdzestien', 'honderdzeventien',
                   'honderdachttien', 'honderdnegentien', 'honderdtwintig', 'honderdeenentwintig',
                   'honderdtweeëntwintig', 'honderddrieëntwintig', 'honderdvierentwintig', 'honderdvijfentwintig',
                   'honderdzesentwintig', 'honderdzevenentwintig', 'honderdachtentwintig', 'honderdnegenentwintig',
                   'honderddertig', 'honderdeenendertig', 'honderdtweeëndertig', 'honderddrieëndertig',
                   'honderdvierendertig', 'honderdvijfendertig', 'honderdzesendertig', 'honderdzevenendertig',
                   'honderdachtendertig', 'honderdnegenendertig', 'honderdveertig', 'honderdeenenveertig',
                   'honderdtweeënveertig', 'honderddrieënveertig', 'honderdvierenveertig', 'honderdvijfenveertig',
                   'honderdzesenveertig', 'honderdzevenenveertig', 'honderdachtenveertig',
                   'honderdnegenenveertig', 'honderdvijftig', 'honderdeenenvijftig', 'honderdtweeënvijftig',
                   'honderddrieënvijftig', 'honderdvierenvijftig', 'honderdvijfenvijftig', 'honderdzesenvijftig',
                   'honderdzevenenvijftig', 'honderdachtenvijftig', 'honderdnegenenvijftig', 'honderdzestig',
                   'honderdeenenzestig', 'honderdtweeënzestig', 'honderddrieënzestig', 'honderdvierenzestig',
                   'honderdvijfenzestig', 'honderdzesenzestig', 'honderdzevenenzestig', 'honderdachtenzestig',
                   'honderdnegenenzestig', 'honderdzeventig', 'honderdeenenzeventig', 'honderdtweeënzeventig',
                   'honderddrieënzeventig', 'honderdvierenzeventig', 'honderdvijfenzeventig', 'honderdzesenzeventig',
                   'honderdzevenenzeventig', 'honderdachtenzeventig', 'honderdnegenenzeventig', 'honderdtachtig',
                   'honderdeenentachtig', 'honderdtweeëntachtig', 'honderddrieëntachtig', 'honderdvierentachtig',
                   'honderdvijfentachtig', 'honderdzesentachtig', 'honderdzevenentachtig', 'honderdachtentachtig',
                   'honderdnegenentachtig', 'honderdnegentig', 'honderdeenennegentig', 'honderdtweeënnegentig',
                   'honderddrieënnegentig', 'honderdvierennegentig', 'honderdvijfennegentig', 'honderdzesennegentig',
                   'honderdzevenennegentig', 'honderdachtennegentig', 'honderdnegenennegentig', 'tweehonderd',
                   'tweehonderdeen', 'tweehonderdtwee', 'tweehonderddrie', 'tweehonderdvier', 'tweehonderdvijf',
                   'tweehonderdzes', 'tweehonderdzeven', 'tweehonderdacht', 'tweehonderdnegen', 'tweehonderdtien',
                   'tweehonderdelf', 'tweehonderdtwaalf', 'tweehonderddertien', 'tweehonderdveertien',
                   'tweehonderdvijftien', 'tweehonderdzestien', 'tweehonderdzeventien', 'tweehonderdachttien',
                   'tweehonderdnegentien', 'tweehonderdtwintig', 'tweehonderdeenentwintig',
                   'tweehonderdtweeëntwintig', 'tweehonderddrieëntwintig', 'tweehonderdvierentwintig',
                   'tweehonderdvijfentwintig', 'tweehonderdzesentwintig', 'tweehonderdzevenentwintig',
                   'tweehonderdachtentwintig', 'tweehonderdnegenentwintig', 'tweehonderddertig',
                   'tweehonderdeenendertig', 'tweehonderdtweeëndertig', 'tweehonderddrieëndertig',
                   'tweehonderdvierendertig', 'tweehonderdvijfendertig', 'tweehonderdzesendertig',
                   'tweehonderdzevenendertig', 'tweehonderdachtendertig', 'tweehonderdnegenendertig',
                   'tweehonderdveertig', 'tweehonderdeenenveertig', 'tweehonderdtweeënveertig',
                   'tweehonderddrieënveertig', 'tweehonderdvierenveertig', 'tweehonderdvijfenveertig',
                   'tweehonderdzesenveertig', 'tweehonderdzevenenveertig', 'tweehonderdachtenveertig',
                   'tweehonderdnegenenveertig', 'tweehonderdvijftig', 'tweehonderdeenenvijftig',
                   'tweehonderdtweeënvijftig', 'tweehonderddrieënvijftig', 'tweehonderdvierenvijftig',
                   'tweehonderdvijfenvijftig', 'tweehonderdzesenvijftig', 'tweehonderdzevenenvijftig',
                   'tweehonderdachtenvijftig', 'tweehonderdnegenenvijftig', 'tweehonderdzestig',
                   'tweehonderdeenenzestig', 'tweehonderdtweeënzestig', 'tweehonderddrieënzestig',
                   'tweehonderdvierenzestig', 'tweehonderdvijfenzestig', 'tweehonderdzesenzestig',
                   'tweehonderdzevenenzestig', 'tweehonderdachtenzestig', 'tweehonderdnegenenzestig',
                   'tweehonderdzeventig', 'tweehonderdeenenzeventig', 'tweehonderdtweeënzeventig',
                   'tweehonderddrieënzeventig', 'tweehonderdvierenzeventig', 'tweehonderdvijfenzeventig',
                   'tweehonderdzesenzeventig', 'tweehonderdzevenenzeventig', 'tweehonderdachtenzeventig',
                   'tweehonderdnegenenzeventig', 'tweehonderdtachtig', 'tweehonderdeenentachtig',
                   'tweehonderdtweeëntachtig', 'tweehonderddrieëntachtig', 'tweehonderdvierentachtig',
                   'tweehonderdvijfentachtig', 'tweehonderdzesentachtig', 'tweehonderdzevenentachtig',
                   'tweehonderdachtentachtig', 'tweehonderdnegenentachtig', 'tweehonderdnegentig',
                   'tweehonderdeenennegentig', 'tweehonderdtweeënnegentig', 'tweehonderddrieënnegentig',
                   'tweehonderdvierennegentig', 'tweehonderdvijfennegentig', 'tweehonderdzesennegentig',
                   'tweehonderdzevenennegentig', 'tweehonderdachtennegentig', 'tweehonderdnegenennegentig',
                   'driehonderd', 'driehonderdeen', 'driehonderdtwee', 'driehonderddrie', 'driehonderdvier',
                   'driehonderdvijf', 'driehonderdzes', 'driehonderdzeven', 'driehonderdacht', 'driehonderdnegen',
                   'driehonderdtien', 'driehonderdelf', 'driehonderdtwaalf', 'driehonderddertien',
                   'driehonderdveertien', 'driehonderdvijftien', 'driehonderdzestien', 'driehonderdzeventien',
                   'driehonderdachttien', 'driehonderdnegentien', 'driehonderdtwintig', 'driehonderdeenentwintig',
                   'driehonderdtweeëntwintig', 'driehonderddrieëntwintig', 'driehonderdvierentwintig',
                   'driehonderdvijfentwintig', 'driehonderdzesentwintig', 'driehonderdzevenentwintig',
                   'driehonderdachtentwintig', 'driehonderdnegenentwintig', 'driehonderddertig',
                   'driehonderdeenendertig', 'driehonderdtweeëndertig', 'driehonderddrieëndertig',
                   'driehonderdvierendertig', 'driehonderdvijfendertig', 'driehonderdzesendertig',
                   'driehonderdzevenendertig', 'driehonderdachtendertig', 'driehonderdnegenendertig',
                   'driehonderdveertig', 'driehonderdeenenveertig', 'driehonderdtweeënveertig',
                   'driehonderddrieënveertig', 'driehonderdvierenveertig', 'driehonderdvijfenveertig',
                   'driehonderdzesenveertig', 'driehonderdzevenenveertig', 'driehonderdachtenveertig',
                   'driehonderdnegenenveertig', 'driehonderdvijftig', 'driehonderdeenenvijftig',
                   'driehonderdtweeënvijftig', 'driehonderddrieënvijftig', 'driehonderdvierenvijftig',
                   'driehonderdvijfenvijftig', 'driehonderdzesenvijftig', 'driehonderdzevenenvijftig',
                   'driehonderdachtenvijftig', 'driehonderdnegenenvijftig', 'driehonderdzestig',
                   'driehonderdeenenzestig', 'driehonderdtweeënzestig', 'driehonderddrieënzestig',
                   'driehonderdvierenzestig', 'driehonderdvijfenzestig', 'driehonderdzesenzestig',
                   'driehonderdzevenenzestig', 'driehonderdachtenzestig', 'driehonderdnegenenzestig',
                   'driehonderdzeventig', 'driehonderdeenenzeventig', 'driehonderdtweeënzeventig',
                   'driehonderddrieënzeventig', 'driehonderdvierenzeventig', 'driehonderdvijfenzeventig',
                   'driehonderdzesenzeventig', 'driehonderdzevenenzeventig', 'driehonderdachtenzeventig',
                   'driehonderdnegenenzeventig', 'driehonderdtachtig', 'driehonderdeenentachtig',
                   'driehonderdtweeëntachtig', 'driehonderddrieëntachtig', 'driehonderdvierentachtig',
                   'driehonderdvijfentachtig', 'driehonderdzesentachtig', 'driehonderdzevenentachtig',
                   'driehonderdachtentachtig', 'driehonderdnegenentachtig', 'driehonderdnegentig',
                   'driehonderdeenennegentig', 'driehonderdtweeënnegentig', 'driehonderddrieënnegentig',
                   'driehonderdvierennegentig', 'driehonderdvijfennegentig', 'driehonderdzesennegentig',
                   'driehonderdzevenennegentig', 'driehonderdachtennegentig', 'driehonderdnegenennegentig',
                   'vierhonderd', 'vierhonderdeen', 'vierhonderdtwee', 'vierhonderddrie', 'vierhonderdvier',
                   'vierhonderdvijf', 'vierhonderdzes', 'vierhonderdzeven', 'vierhonderdacht', 'vierhonderdnegen',
                   'vierhonderdtien', 'vierhonderdelf', 'vierhonderdtwaalf', 'vierhonderddertien',
                   'vierhonderdveertien',
                   'vierhonderdvijftien', 'vierhonderdzestien', 'vierhonderdzeventien', 'vierhonderdachttien',
                   'vierhonderdnegentien', 'vierhonderdtwintig', 'vierhonderdeenentwintig', 'vierhonderdtweeëntwintig',
                   'vierhonderddrieëntwintig', 'vierhonderdvierentwintig', 'vierhonderdvijfentwintig',
                   'vierhonderdzesentwintig', 'vierhonderdzevenentwintig', 'vierhonderdachtentwintig',
                   'vierhonderdnegenentwintig', 'vierhonderddertig', 'vierhonderdeenendertig',
                   'vierhonderdtweeëndertig',
                   'vierhonderddrieëndertig', 'vierhonderdvierendertig', 'vierhonderdvijfendertig',
                   'vierhonderdzesendertig', 'vierhonderdzevenendertig', 'vierhonderdachtendertig',
                   'vierhonderdnegenendertig', 'vierhonderdveertig', 'vierhonderdeenenveertig',
                   'vierhonderdtweeënveertig', 'vierhonderddrieënveertig', 'vierhonderdvierenveertig',
                   'vierhonderdvijfenveertig', 'vierhonderdzesenveertig', 'vierhonderdzevenenveertig',
                   'vierhonderdachtenveertig',
                   'vierhonderdnegenenveertig', 'vierhonderdvijftig', 'vierhonderdeenenvijftig',
                   'vierhonderdtweeënvijftig',
                   'vierhonderddrieënvijftig', 'vierhonderdvierenvijftig', 'vierhonderdvijfenvijftig',
                   'vierhonderdzesenvijftig',
                   'vierhonderdzevenenvijftig', 'vierhonderdachtenvijftig', 'vierhonderdnegenenvijftig',
                   'vierhonderdzestig',
                   'vierhonderdeenenzestig', 'vierhonderdtweeënzestig', 'vierhonderddrieënzestig',
                   'vierhonderdvierenzestig',
                   'vierhonderdvijfenzestig', 'vierhonderdzesenzestig', 'vierhonderdzevenenzestig',
                   'vierhonderdachtenzestig',
                   'vierhonderdnegenenzestig', 'vierhonderdzeventig', 'vierhonderdeenenzeventig',
                   'vierhonderdtweeënzeventig',
                   'vierhonderddrieënzeventig', 'vierhonderdvierenzeventig', 'vierhonderdvijfenzeventig',
                   'vierhonderdzesenzeventig',
                   'vierhonderdzevenenzeventig', 'vierhonderdachtenzeventig', 'vierhonderdnegenenzeventig',
                   'vierhonderdtachtig',
                   'vierhonderdeenentachtig', 'vierhonderdtweeëntachtig', 'vierhonderddrieëntachtig',
                   'vierhonderdvierentachtig',
                   'vierhonderdvijfentachtig', 'vierhonderdzesentachtig', 'vierhonderdzevenentachtig',
                   'vierhonderdachtentachtig',
                   'vierhonderdnegenentachtig', 'vierhonderdnegentig', 'vierhonderdeenennegentig',
                   'vierhonderdtweeënnegentig',
                   'vierhonderddrieënnegentig', 'vierhonderdvierennegentig', 'vierhonderdvijfennegentig',
                   'vierhonderdzesennegentig',
                   'vierhonderdzevenennegentig', 'vierhonderdachtennegentig', 'vierhonderdnegenennegentig',
                   'vijfhonderd',
                   'vijfhonderdeen',
                   'vijfhonderdtwee', 'vijfhonderddrie', 'vijfhonderdvier', 'vijfhonderdvijf', 'vijfhonderdzes',
                   'vijfhonderdzeven',
                   'vijfhonderdacht', 'vijfhonderdnegen', 'vijfhonderdtien', 'vijfhonderdelf', 'vijfhonderdtwaalf',
                   'vijfhonderddertien',
                   'vijfhonderdveertien', 'vijfhonderdvijftien', 'vijfhonderdzestien', 'vijfhonderdzeventien',
                   'vijfhonderdachttien',
                   'vijfhonderdnegentien', 'vijfhonderdtwintig', 'vijfhonderdeenentwintig', 'vijfhonderdtweeëntwintig',
                   'vijfhonderddrieëntwintig', 'vijfhonderdvierentwintig', 'vijfhonderdvijfentwintig',
                   'vijfhonderdzesentwintig',
                   'vijfhonderdzevenentwintig', 'vijfhonderdachtentwintig', 'vijfhonderdnegenentwintig',
                   'vijfhonderddertig',
                   'vijfhonderdeenendertig', 'vijfhonderdtweeëndertig', 'vijfhonderddrieëndertig',
                   'vijfhonderdvierendertig',
                   'vijfhonderdvijfendertig', 'vijfhonderdzesendertig', 'vijfhonderdzevenendertig',
                   'vijfhonderdachtendertig',
                   'vijfhonderdnegenendertig', 'vijfhonderdveertig', 'vijfhonderdeenenveertig',
                   'vijfhonderdtweeënveertig',
                   'vijfhonderddrieënveertig', 'vijfhonderdvierenveertig', 'vijfhonderdvijfenveertig',
                   'vijfhonderdzesenveertig',
                   'vijfhonderdzevenenveertig', 'vijfhonderdachtenveertig', 'vijfhonderdnegenenveertig',
                   'vijfhonderdvijftig',
                   'vijfhonderdeenenvijftig', 'vijfhonderdtweeënvijftig', 'vijfhonderddrieënvijftig',
                   'vijfhonderdvierenvijftig',
                   'vijfhonderdvijfenvijftig', 'vijfhonderdzesenvijftig', 'vijfhonderdzevenenvijftig',
                   'vijfhonderdachtenvijftig',
                   'vijfhonderdnegenenvijftig', 'vijfhonderdzestig', 'vijfhonderdeenenzestig',
                   'vijfhonderdtweeënzestig',
                   'vijfhonderddrieënzestig', 'vijfhonderdvierenzestig', 'vijfhonderdvijfenzestig',
                   'vijfhonderdzesenzestig',
                   'vijfhonderdzevenenzestig', 'vijfhonderdachtenzestig', 'vijfhonderdnegenenzestig',
                   'vijfhonderdzeventig',
                   'vijfhonderdeenenzeventig', 'vijfhonderdtweeënzeventig', 'vijfhonderddrieënzeventig',
                   'vijfhonderdvierenzeventig',
                   'vijfhonderdvijfenzeventig', 'vijfhonderdzesenzeventig', 'vijfhonderdzevenenzeventig',
                   'vijfhonderdachtenzeventig',
                   'vijfhonderdnegenenzeventig', 'vijfhonderdtachtig', 'vijfhonderdeenentachtig',
                   'vijfhonderdtweeëntachtig',
                   'vijfhonderddrieëntachtig', 'vijfhonderdvierentachtig', 'vijfhonderdvijfentachtig',
                   'vijfhonderdzesentachtig',
                   'vijfhonderdzevenentachtig', 'vijfhonderdachtentachtig', 'vijfhonderdnegenentachtig',
                   'vijfhonderdnegentig',
                   'vijfhonderdeenennegentig', 'vijfhonderdtweeënnegentig', 'vijfhonderddrieënnegentig',
                   'vijfhonderdvierennegentig',
                   'vijfhonderdvijfennegentig', 'vijfhonderdzesennegentig', 'vijfhonderdzevenennegentig',
                   'vijfhonderdachtennegentig',
                   'vijfhonderdnegenennegentig', 'zeshonderd', 'zeshonderdeen', 'zeshonderdtwee', 'zeshonderddrie',
                   'zeshonderdvier',
                   'zeshonderdvijf', 'zeshonderdzes', 'zeshonderdzeven', 'zeshonderdacht', 'zeshonderdnegen',
                   'zeshonderdtien',
                   'zeshonderdelf',
                   'zeshonderdtwaalf', 'zeshonderddertien', 'zeshonderdveertien', 'zeshonderdvijftien',
                   'zeshonderdzestien',
                   'zeshonderdzeventien', 'zeshonderdachttien', 'zeshonderdnegentien', 'zeshonderdtwintig',
                   'zeshonderdeenentwintig',
                   'zeshonderdtweeëntwintig', 'zeshonderddrieëntwintig', 'zeshonderdvierentwintig',
                   'zeshonderdvijfentwintig',
                   'zeshonderdzesentwintig', 'zeshonderdzevenentwintig', 'zeshonderdachtentwintig',
                   'zeshonderdnegenentwintig',
                   'zeshonderddertig', 'zeshonderdeenendertig', 'zeshonderdtweeëndertig', 'zeshonderddrieëndertig',
                   'zeshonderdvierendertig',
                   'zeshonderdvijfendertig', 'zeshonderdzesendertig', 'zeshonderdzevenendertig',
                   'zeshonderdachtendertig',
                   'zeshonderdnegenendertig', 'zeshonderdveertig', 'zeshonderdeenenveertig', 'zeshonderdtweeënveertig',
                   'zeshonderddrieënveertig', 'zeshonderdvierenveertig', 'zeshonderdvijfenveertig',
                   'zeshonderdzesenveertig',
                   'zeshonderdzevenenveertig', 'zeshonderdachtenveertig', 'zeshonderdnegenenveertig',
                   'zeshonderdvijftig',
                   'zeshonderdeenenvijftig', 'zeshonderdtweeënvijftig', 'zeshonderddrieënvijftig',
                   'zeshonderdvierenvijftig',
                   'zeshonderdvijfenvijftig', 'zeshonderdzesenvijftig', 'zeshonderdzevenenvijftig',
                   'zeshonderdachtenvijftig',
                   'zeshonderdnegenenvijftig', 'zeshonderdzestig', 'zeshonderdeenenzestig', 'zeshonderdtweeënzestig',
                   'zeshonderddrieënzestig',
                   'zeshonderdvierenzestig', 'zeshonderdvijfenzestig', 'zeshonderdzesenzestig',
                   'zeshonderdzevenenzestig',
                   'zeshonderdachtenzestig', 'zeshonderdnegenenzestig', 'zeshonderdzeventig', 'zeshonderdeenenzeventig',
                   'zeshonderdtweeënzeventig', 'zeshonderddrieënzeventig', 'zeshonderdvierenzeventig',
                   'zeshonderdvijfenzeventig',
                   'zeshonderdzesenzeventig', 'zeshonderdzevenenzeventig', 'zeshonderdachtenzeventig',
                   'zeshonderdnegenenzeventig',
                   'zeshonderdtachtig', 'zeshonderdeenentachtig', 'zeshonderdtweeëntachtig', 'zeshonderddrieëntachtig',
                   'zeshonderdvierentachtig', 'zeshonderdvijfentachtig', 'zeshonderdzesentachtig',
                   'zeshonderdzevenentachtig',
                   'zeshonderdachtentachtig', 'zeshonderdnegenentachtig', 'zeshonderdnegentig',
                   'zeshonderdeenennegentig',
                   'zeshonderdtweeënnegentig', 'zeshonderddrieënnegentig', 'zeshonderdvierennegentig',
                   'zeshonderdvijfennegentig',
                   'zeshonderdzesennegentig', 'zeshonderdzevenennegentig', 'zeshonderdachtennegentig',
                   'zeshonderdnegenennegentig',
                   'zevenhonderd', 'zevenhonderdeen', 'zevenhonderdtwee', 'zevenhonderddrie', 'zevenhonderdvier',
                   'zevenhonderdvijf',
                   'zevenhonderdzes', 'zevenhonderdzeven', 'zevenhonderdacht', 'zevenhonderdnegen', 'zevenhonderdtien',
                   'zevenhonderdelf',
                   'zevenhonderdtwaalf', 'zevenhonderddertien', 'zevenhonderdveertien', 'zevenhonderdvijftien',
                   'zevenhonderdzestien',
                   'zevenhonderdzeventien', 'zevenhonderdachttien', 'zevenhonderdnegentien', 'zevenhonderdtwintig',
                   'zevenhonderdeenentwintig',
                   'zevenhonderdtweeëntwintig', 'zevenhonderddrieëntwintig', 'zevenhonderdvierentwintig',
                   'zevenhonderdvijfentwintig',
                   'zevenhonderdzesentwintig', 'zevenhonderdzevenentwintig', 'zevenhonderdachtentwintig',
                   'zevenhonderdnegenentwintig',
                   'zevenhonderddertig', 'zevenhonderdeenendertig', 'zevenhonderdtweeëndertig',
                   'zevenhonderddrieëndertig',
                   'zevenhonderdvierendertig', 'zevenhonderdvijfendertig', 'zevenhonderdzesendertig',
                   'zevenhonderdzevenendertig',
                   'zevenhonderdachtendertig', 'zevenhonderdnegenendertig', 'zevenhonderdveertig',
                   'zevenhonderdeenenveertig',
                   'zevenhonderdtweeënveertig', 'zevenhonderddrieënveertig', 'zevenhonderdvierenveertig',
                   'zevenhonderdvijfenveertig',
                   'zevenhonderdzesenveertig', 'zevenhonderdzevenenveertig', 'zevenhonderdachtenveertig',
                   'zevenhonderdnegenenveertig',
                   'zevenhonderdvijftig', 'zevenhonderdeenenvijftig', 'zevenhonderdtweeënvijftig',
                   'zevenhonderddrieënvijftig',
                   'zevenhonderdvierenvijftig', 'zevenhonderdvijfenvijftig', 'zevenhonderdzesenvijftig',
                   'zevenhonderdzevenenvijftig',
                   'zevenhonderdachtenvijftig', 'zevenhonderdnegenenvijftig', 'zevenhonderdzestig',
                   'zevenhonderdeenenzestig',
                   'zevenhonderdtweeënzestig', 'zevenhonderddrieënzestig', 'zevenhonderdvierenzestig',
                   'zevenhonderdvijfenzestig',
                   'zevenhonderdzesenzestig', 'zevenhonderdzevenenzestig', 'zevenhonderdachtenzestig',
                   'zevenhonderdnegenenzestig',
                   'zevenhonderdzeventig', 'zevenhonderdeenenzeventig', 'zevenhonderdtweeënzeventig',
                   'zevenhonderddrieënzeventig',
                   'zevenhonderdvierenzeventig', 'zevenhonderdvijfenzeventig', 'zevenhonderdzesenzeventig',
                   'zevenhonderdzevenenzeventig',
                   'zevenhonderdachtenzeventig', 'zevenhonderdnegenenzeventig', 'zevenhonderdtachtig',
                   'zevenhonderdeenentachtig',
                   'zevenhonderdtweeëntachtig', 'zevenhonderddrieëntachtig', 'zevenhonderdvierentachtig',
                   'zevenhonderdvijfentachtig',
                   'zevenhonderdzesentachtig', 'zevenhonderdzevenentachtig', 'zevenhonderdachtentachtig',
                   'zevenhonderdnegenentachtig',
                   'zevenhonderdnegentig', 'zevenhonderdeenennegentig', 'zevenhonderdtweeënnegentig',
                   'zevenhonderddrieënnegentig',
                   'zevenhonderdvierennegentig', 'zevenhonderdvijfennegentig', 'zevenhonderdzesennegentig',
                   'zevenhonderdzevenennegentig',
                   'zevenhonderdachtennegentig', 'zevenhonderdnegenennegentig', 'achthonderd', 'achthonderdeen',
                   'achthonderdtwee',
                   'achthonderddrie', 'achthonderdvier', 'achthonderdvijf', 'achthonderdzes', 'achthonderdzeven',
                   'achthonderdacht',
                   'achthonderdnegen', 'achthonderdtien', 'achthonderdelf', 'achthonderdtwaalf', 'achthonderddertien',
                   'achthonderdveertien',
                   'achthonderdvijftien', 'achthonderdzestien', 'achthonderdzeventien', 'achthonderdachttien',
                   'achthonderdnegentien',
                   'achthonderdtwintig', 'achthonderdeenentwintig', 'achthonderdtweeëntwintig',
                   'achthonderddrieëntwintig',
                   'achthonderdvierentwintig', 'achthonderdvijfentwintig', 'achthonderdzesentwintig',
                   'achthonderdzevenentwintig',
                   'achthonderdachtentwintig', 'achthonderdnegenentwintig', 'achthonderddertig',
                   'achthonderdeenendertig',
                   'achthonderdtweeëndertig', 'achthonderddrieëndertig', 'achthonderdvierendertig',
                   'achthonderdvijfendertig',
                   'achthonderdzesendertig', 'achthonderdzevenendertig', 'achthonderdachtendertig',
                   'achthonderdnegenendertig',
                   'achthonderdveertig', 'achthonderdeenenveertig', 'achthonderdtweeënveertig',
                   'achthonderddrieënveertig',
                   'achthonderdvierenveertig', 'achthonderdvijfenveertig', 'achthonderdzesenveertig',
                   'achthonderdzevenenveertig',
                   'achthonderdachtenveertig', 'achthonderdnegenenveertig', 'achthonderdvijftig',
                   'achthonderdeenenvijftig',
                   'achthonderdtweeënvijftig', 'achthonderddrieënvijftig', 'achthonderdvierenvijftig',
                   'achthonderdvijfenvijftig',
                   'achthonderdzesenvijftig', 'achthonderdzevenenvijftig', 'achthonderdachtenvijftig',
                   'achthonderdnegenenvijftig',
                   'achthonderdzestig', 'achthonderdeenenzestig', 'achthonderdtweeënzestig', 'achthonderddrieënzestig',
                   'achthonderdvierenzestig', 'achthonderdvijfenzestig', 'achthonderdzesenzestig',
                   'achthonderdzevenenzestig',
                   'achthonderdachtenzestig', 'achthonderdnegenenzestig', 'achthonderdzeventig',
                   'achthonderdeenenzeventig',
                   'achthonderdtweeënzeventig', 'achthonderddrieënzeventig', 'achthonderdvierenzeventig',
                   'achthonderdvijfenzeventig',
                   'achthonderdzesenzeventig', 'achthonderdzevenenzeventig', 'achthonderdachtenzeventig',
                   'achthonderdnegenenzeventig',
                   'achthonderdtachtig', 'achthonderdeenentachtig', 'achthonderdtweeëntachtig',
                   'achthonderddrieëntachtig',
                   'achthonderdvierentachtig', 'achthonderdvijfentachtig', 'achthonderdzesentachtig',
                   'achthonderdzevenentachtig',
                   'achthonderdachtentachtig', 'achthonderdnegenentachtig', 'achthonderdnegentig',
                   'achthonderdeenennegentig',
                   'achthonderdtweeënnegentig', 'achthonderddrieënnegentig', 'achthonderdvierennegentig',
                   'achthonderdvijfennegentig',
                   'achthonderdzesennegentig', 'achthonderdzevenennegentig', 'achthonderdachtennegentig',
                   'achthonderdnegenennegentig',
                   'negenhonderd', 'negenhonderdeen', 'negenhonderdtwee', 'negenhonderddrie', 'negenhonderdvier',
                   'negenhonderdvijf',
                   'negenhonderdzes', 'negenhonderdzeven', 'negenhonderdacht', 'negenhonderdnegen', 'negenhonderdtien',
                   'negenhonderdelf',
                   'negenhonderdtwaalf', 'negenhonderddertien', 'negenhonderdveertien', 'negenhonderdvijftien',
                   'negenhonderdzestien',
                   'negenhonderdzeventien', 'negenhonderdachttien', 'negenhonderdnegentien', 'negenhonderdtwintig',
                   'negenhonderdeenentwintig',
                   'negenhonderdtweeëntwintig', 'negenhonderddrieëntwintig', 'negenhonderdvierentwintig',
                   'negenhonderdvijfentwintig',
                   'negenhonderdzesentwintig', 'negenhonderdzevenentwintig', 'negenhonderdachtentwintig',
                   'negenhonderdnegenentwintig',
                   'negenhonderddertig', 'negenhonderdeenendertig', 'negenhonderdtweeëndertig',
                   'negenhonderddrieëndertig',
                   'negenhonderdvierendertig', 'negenhonderdvijfendertig', 'negenhonderdzesendertig',
                   'negenhonderdzevenendertig',
                   'negenhonderdachtendertig', 'negenhonderdnegenendertig', 'negenhonderdveertig',
                   'negenhonderdeenenveertig',
                   'negenhonderdtweeënveertig',
                   'negenhonderddrieënveertig', 'negenhonderdvierenveertig', 'negenhonderdvijfenveertig',
                   'negenhonderdzesenveertig',
                   'negenhonderdzevenenveertig', 'negenhonderdachtenveertig', 'negenhonderdnegenenveertig',
                   'negenhonderdvijftig',
                   'negenhonderdeenenvijftig', 'negenhonderdtweeënvijftig', 'negenhonderddrieënvijftig',
                   'negenhonderdvierenvijftig',
                   'negenhonderdvijfenvijftig', 'negenhonderdzesenvijftig', 'negenhonderdzevenenvijftig',
                   'negenhonderdachtenvijftig',
                   'negenhonderdnegenenvijftig', 'negenhonderdzestig', 'negenhonderdeenenzestig',
                   'negenhonderdtweeënzestig',
                   'negenhonderddrieënzestig', 'negenhonderdvierenzestig', 'negenhonderdvijfenzestig',
                   'negenhonderdzesenzestig',
                   'negenhonderdzevenenzestig', 'negenhonderdachtenzestig', 'negenhonderdnegenenzestig',
                   'negenhonderdzeventig',
                   'negenhonderdeenenzeventig', 'negenhonderdtweeënzeventig', 'negenhonderddrieënzeventig',
                   'negenhonderdvierenzeventig',
                   'negenhonderdvijfenzeventig', 'negenhonderdzesenzeventig', 'negenhonderdzevenenzeventig',
                   'negenhonderdachtenzeventig',
                   'negenhonderdnegenenzeventig', 'negenhonderdtachtig', 'negenhonderdeenentachtig',
                   'negenhonderdtweeëntachtig',
                   'negenhonderddrieëntachtig', 'negenhonderdvierentachtig', 'negenhonderdvijfentachtig',
                   'negenhonderdzesentachtig',
                   'negenhonderdzevenentachtig', 'negenhonderdachtentachtig', 'negenhonderdnegenentachtig',
                   'negenhonderdnegentig',
                   'negenhonderdeenennegentig', 'negenhonderdtweeënnegentig', 'negenhonderddrieënnegentig',
                   'negenhonderdvierennegentig',
                   'negenhonderdvijfennegentig', 'negenhonderdzesennegentig', 'negenhonderdzevenennegentig',
                   'negenhonderdachtennegentig')
####################################################################################################################
quantifiers = ('verschillende', 'enkele', 'meerdere')  # have a different/no meaning without the '-e'
comparison_with_separate_sign = ('later', 'alle', 'beter', 'best', 'minder', 'meer', 'meest')
comparison_with_separate_sign_with_e_del_e = ('betere', 'meeste')  # only 'beter' in gloss_id
comparison_with_separate_sign_with_e_keep_e = ('laatste', 'beste', 'voorlaatste')  # 'laatste', 'beste' are in gloss_id
comparison_with_separate_sign_with_e_add_e = ('laatst', 'best', 'voorlaatst')
other_exeptions = ('vermist', 'volgend', 'volgende')
new_form_is_text = comparison_with_separate_sign_with_e_keep_e + comparison_with_separate_sign + other_exeptions \
                   + quantifiers
####################################################################################################################
assigned_gloss_ids = ('GELIJKEN-OP', 'WG', 'VAN-3', 'GA-NAAR')
####################################################################################################################
options_before = ('dubbele articulatie', 'VEEL ', 'VERSCHILLENDE ', 'ENKELE ', 'ALLEMAAL ', 'ERG-VEEL ',
                  'sweep ', 'wg+++ ', '')
# other options, but often not correct:  'HOOP ', 'RIJ ', 'GROEP ', 'PIRAMIDE ', 'BERG ', 'KETTING ', 'ROMMEL ',
options_after = (' WG-6', ' sweep', '+++', ' wg+++', '')  # classifier constructie ' hoop',
# also an empty one because plural is not always expressed
####################################################################################################################
and_or = ('en', 'of')
more_less = ("meer", 'minder')
what = ('wat',)
who = ('wie',)
that = ('die', 'dat')
tw_to_ignore = ('hoeveel', 'zoveel')
words_without_gloss = ('het', 'wel', 'die', 'dat', 'deze', 'dit', 'elkaar', 'toch', 'zich', "’", 'daar', 'hoor')
start_end = ('eind', 'begin')
time_back = ('geleden',)
elements_incorporated_in_time_sign = ('dit', 'deze', 'elke', 'elk', 'vorig', 'vorige')
general_there = ('er',)
not_ = ('niet',)
####################################################################################################################
gloss_ids_json = read_file('gloss_ids.json')
####################################################################################################################
verb_with_prep = read_file('verb_with_prep.json')

