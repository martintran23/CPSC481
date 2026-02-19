from probability4e import *
T, F = True, False


class Diagnostics:
    """ Use a Bayesian network to diagnose between three lung diseases """

    def __init__(self):

        self.bn = BayesNet([

            # Priors
            ('VisitToAsia', '', 0.01),
            ('Smoking', '', 0.5),

            # TB depends on VisitToAsia
            ('TB', 'VisitToAsia',
             {True: 0.05, False: 0.01}),

            # Cancer probabilities
            ('Cancer', 'Smoking',
             {True: 0.1, False: 0.01}),

            # Bronchitis depends on Smoking
            ('Bronchitis', 'Smoking',
             {True: 0.6, False: 0.3}),

            # Deterministic OR
            ('TBorCancer', 'TB Cancer',
             {(True, True): 1.0,
              (True, False): 1.0,
              (False, True): 1.0,
              (False, False): 0.0}),

            # Xray depends on TBorCancer
            ('Xray', 'TBorCancer',
             {True: 0.98, False: 0.05}),

            # Dyspnea depends on TBorCancer and Bronchitis
            ('Dyspnea', 'TBorCancer Bronchitis',
             {(True, True): 0.9,
              (True, False): 0.7,
              (False, True): 0.8,
              (False, False): 0.1})
        ])

    def diagnose(self, asia, smoking, xray, dyspnea):

        evidence = {}

        # Visit to Asia
        if asia == "Yes":
            evidence['VisitToAsia'] = True
        elif asia == "No":
            evidence['VisitToAsia'] = False

        # Smoking
        if smoking == "Yes":
            evidence['Smoking'] = True
        elif smoking == "No":
            evidence['Smoking'] = False

        # Xray
        if xray == "Abnormal":
            evidence['Xray'] = True
        elif xray == "Normal":
            evidence['Xray'] = False

        # Dyspnea
        if dyspnea == "Present":
            evidence['Dyspnea'] = True
        elif dyspnea == "Absent":
            evidence['Dyspnea'] = False

        # Inference
        tb_prob = enumeration_ask('TB', evidence, self.bn)[True]
        cancer_prob = enumeration_ask('Cancer', evidence, self.bn)[True]
        bronchitis_prob = enumeration_ask('Bronchitis', evidence, self.bn)[True]

        probs = {
            "TB": tb_prob,
            "Cancer": cancer_prob,
            "Bronchitis": bronchitis_prob
        }

        best_disease = max(probs, key=probs.get)

        return [best_disease, round(probs[best_disease], 3)]
