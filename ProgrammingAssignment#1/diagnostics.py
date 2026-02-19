from probability4e import *
T, F = True, False

class Diagnostics:
    """ Use a Bayesian network to diagnose between three lung diseases """
    def __init__(self):

        # Build the Bayesian Network
        self.bn = BayesNet([

            # Priors
            ('VisitToAsia', '', 0.01),
            ('Smoking', '', 0.5),

            # Tuberculosis depends on VisitToAsia
            ('TB', 'VisitToAsia',
             {T: 0.05, F: 0.01}),

            # ✅ Correct Cancer probabilities (PSA fix)
            ('Cancer', 'Smoking',
             {T: 0.1, F: 0.01}),

            # Bronchitis depends on Smoking
            ('Bronchitis', 'Smoking',
             {T: 0.6, F: 0.3}),

            # Deterministic OR node
            ('TBorCancer', 'TB Cancer',
             {(T, T): 1.0,
              (T, F): 1.0,
              (F, T): 1.0,
              (F, F): 0.0}),

            # Xray depends on TBorCancer
            ('Xray', 'TBorCancer',
             {T: 0.98, F: 0.05}),

            # Dyspnea depends on TBorCancer and Bronchitis
            ('Dyspnea', 'TBorCancer Bronchitis',
             {(T, T): 0.9,
              (T, F): 0.7,
              (F, T): 0.8,
              (F, F): 0.1})
        ])

    def diagnose(self, asia, smoking, xray, dyspnea):

        evidence = {}

        # Convert Visit to Asia
        if asia == "Yes":
            evidence['VisitToAsia'] = T
        elif asia == "No":
            evidence['VisitToAsia'] = F

        # Convert Smoking
        if smoking == "Yes":
            evidence['Smoking'] = T
        elif smoking == "No":
            evidence['Smoking'] = F

        # Convert Xray
        if xray == "Abnormal":
            evidence['Xray'] = T
        elif xray == "Normal":
            evidence['Xray'] = F

        # Convert Dyspnea
        if dyspnea == "Present":
            evidence['Dyspnea'] = T
        elif dyspnea == "Absent":
            evidence['Dyspnea'] = F

        # Compute probabilities using enumeration
        tb_prob = enumeration_ask('TB', evidence, self.bn)[T]
        cancer_prob = enumeration_ask('Cancer', evidence, self.bn)[T]
        bronchitis_prob = enumeration_ask('Bronchitis', evidence, self.bn)[T]

        # Determine most likely disease
        probs = {
            "TB": tb_prob,
            "Cancer": cancer_prob,
            "Bronchitis": bronchitis_prob
        }

        best_disease = max(probs, key=probs.get)

        return [best_disease, probs[best_disease]]
