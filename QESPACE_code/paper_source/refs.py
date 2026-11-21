"""Reference database. status: 'ok' = standard reference recalled with confidence,
'src' = taken from the author's Q-ESPACE source document (not independently checked),
'chk' = added during drafting, details must be checked against the DOI before submission."""
R = {}


def add(key, authors, title, journal, vol="", pages="", year="", doi="", status="ok"):
    R[key] = dict(authors=authors, title=title, journal=journal, vol=vol, pages=pages, year=year, doi=doi, status=status)


add("attwell2001", "Attwell, D. & Laughlin, S. B.", "An energy budget for signaling in the grey matter of the brain", "J. Cereb. Blood Flow Metab.", "21", "1133--1145", "2001", "10.1097/00004647-200110000-00001")
add("harris2012", "Harris, J. J., Jolivet, R. & Attwell, D.", "Synaptic energy use and supply", "Neuron", "75", "762--777", "2012", "10.1016/j.neuron.2012.08.019")
add("levy2002", "Levy, W. B. & Baxter, R. A.", "Energy-efficient neuronal computation via quantal synaptic failures", "J. Neurosci.", "22", "4746--4755", "2002", "10.1523/JNEUROSCI.22-11-04746.2002")
add("placais2013", "Pla\\c{c}ais, P.-Y. & Preat, T.", "To favor survival under food shortage, the brain disables costly memory", "Science", "339", "440--442", "2013", "10.1126/science.1226018")
add("li2020", "Li, H. L. & van Rossum, M. C. W.", "Energy efficient synaptic plasticity", "eLife", "9", "e50804", "2020", "10.7554/eLife.50804")
add("malkin2024", "Malkin, J., O'Donnell, C., Houghton, C. J. & Aitchison, L.", "Signatures of Bayesian inference emerge from energy-efficient synapses", "eLife", "12", "RP92595", "2024", "10.7554/eLife.92595", "src")
add("schug2021", "Schug, S., Benzing, F. & Steger, A.", "Presynaptic stochasticity improves energy efficiency and helps alleviate the stability-plasticity dilemma", "eLife", "10", "e69884", "2021", "10.7554/eLife.69884", "src")
add("kappel2015", "Kappel, D., Habenschuss, S., Legenstein, R. & Maass, W.", "Network plasticity as Bayesian inference", "PLoS Comput. Biol.", "11", "e1004485", "2015", "10.1371/journal.pcbi.1004485")
add("fremaux2016", "Fr\\'emaux, N. & Gerstner, W.", "Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules", "Front. Neural Circuits", "9", "85", "2016", "10.3389/fncir.2015.00085")
add("bellec2020", "Bellec, G. et al.", "A solution to the learning dilemma for recurrent networks of spiking neurons", "Nat. Commun.", "11", "3625", "2020", "10.1038/s41467-020-17236-y")
add("neftci2019", "Neftci, E. O., Mostafa, H. & Zenke, F.", "Surrogate gradient learning in spiking neural networks", "IEEE Signal Process. Mag.", "36", "51--63", "2019", "10.1109/MSP.2019.2931595")
add("kirkpatrick2017", "Kirkpatrick, J. et al.", "Overcoming catastrophic forgetting in neural networks", "Proc. Natl Acad. Sci. USA", "114", "3521--3526", "2017", "10.1073/pnas.1611835114")
add("zenke2017", "Zenke, F., Poole, B. & Ganguli, S.", "Continual learning through synaptic intelligence", "Proc. Mach. Learn. Res.", "70", "3987--3995", "2017", "")
add("fusi2005", "Fusi, S., Drew, P. J. & Abbott, L. F.", "Cascade models of synaptically stored memories", "Neuron", "45", "599--611", "2005", "10.1016/j.neuron.2005.02.001")
add("benna2016", "Benna, M. K. & Fusi, S.", "Computational principles of synaptic memory consolidation", "Nat. Neurosci.", "19", "1697--1706", "2016", "10.1038/nn.4401")
add("frey1997", "Frey, U. & Morris, R. G. M.", "Synaptic tagging and long-term potentiation", "Nature", "385", "533--536", "1997", "10.1038/385533a0")
add("poirazi2020", "Poirazi, P. & Papoutsi, A.", "Illuminating dendritic function with computational models", "Nat. Rev. Neurosci.", "21", "303--321", "2020", "10.1038/s41583-020-0301-7")
add("roy2019", "Roy, K., Jaiswal, A. & Panda, P.", "Towards spike-based machine intelligence with neuromorphic computing", "Nature", "575", "607--617", "2019", "10.1038/s41586-019-1677-2")
add("davies2018", "Davies, M. et al.", "Loihi: a neuromorphic manycore processor with on-chip learning", "IEEE Micro", "38", "82--99", "2018", "10.1109/MM.2018.112130359")
add("markovic2020", "Markovi\\'c, D. & Grollier, J.", "Quantum neuromorphic computing", "Appl. Phys. Lett.", "117", "150501", "2020", "10.1063/5.0020014")
add("tegmark2000", "Tegmark, M.", "Importance of quantum decoherence in brain processes", "Phys. Rev. E", "61", "4194--4206", "2000", "10.1103/PhysRevE.61.4194")
add("fisher2015", "Fisher, M. P. A.", "Quantum cognition: the possibility of processing with nuclear spins in the brain", "Ann. Phys.", "362", "593--602", "2015", "10.1016/j.aop.2015.08.020")
add("preskill2018", "Preskill, J.", "Quantum computing in the NISQ era and beyond", "Quantum", "2", "79", "2018", "10.22331/q-2018-08-06-79")
add("farhi2014", "Farhi, E., Goldstone, J. & Gutmann, S.", "A quantum approximate optimization algorithm", "Preprint at https://arxiv.org/abs/1411.4028", "", "", "2014", "")
add("blekos2024", "Blekos, K. et al.", "A review on quantum approximate optimization algorithm and its variants", "Phys. Rep.", "1068", "1--66", "2024", "10.1016/j.physrep.2024.03.002", "src")
add("hadfield2019", "Hadfield, S. et al.", "From the quantum approximate optimization algorithm to a quantum alternating operator ansatz", "Algorithms", "12", "34", "2019", "10.3390/a12020034")
add("lucas2014", "Lucas, A.", "Ising formulations of many NP problems", "Front. Phys.", "2", "5", "2014", "10.3389/fphy.2014.00005")
add("pisinger2007", "Pisinger, D.", "The quadratic knapsack problem---a survey", "Discrete Appl. Math.", "155", "623--648", "2007", "10.1016/j.dam.2006.08.007")
add("montanez2024", "Monta\\~nez-Barrera, J. A., Willsch, D., Maldonado-Romo, A. & Michielsen, K.", "Unbalanced penalization: a new approach to encode inequality constraints of combinatorial problems for quantum optimization algorithms", "Quantum Sci. Technol.", "9", "025022", "2024", "10.1088/2058-9565/ad35e4", "chk")
add("zhou2020", "Zhou, L., Wang, S.-T., Choi, S., Pichler, H. & Lukin, M. D.", "Quantum approximate optimization algorithm: performance, mechanism, and implementation on near-term devices", "Phys. Rev. X", "10", "021067", "2020", "10.1103/PhysRevX.10.021067")
add("barkoutsos2020", "Barkoutsos, P. K. et al.", "Improving variational quantum optimization using CVaR", "Quantum", "4", "256", "2020", "10.22331/q-2020-04-20-256")
add("weidenfeller2022", "Weidenfeller, J. et al.", "Scaling of the quantum approximate optimization algorithm on superconducting qubit based hardware", "Quantum", "6", "870", "2022", "10.22331/q-2022-12-07-870")
add("abbas2024", "Abbas, A. et al.", "Challenges and opportunities in quantum optimization", "Nat. Rev. Phys.", "6", "718--735", "2024", "10.1038/s42254-024-00770-9")
add("koch2026", "Koch, T. et al.", "The quantum optimization benchmarking library", "Nat. Comput. Sci.", "", "", "2026", "10.1038/s43588-026-00991-1", "src")
add("temme2017", "Temme, K., Bravyi, S. & Gambetta, J. M.", "Error mitigation for short-depth quantum circuits", "Phys. Rev. Lett.", "119", "180509", "2017", "10.1103/PhysRevLett.119.180509")
add("li2017", "Li, Y. & Benjamin, S. C.", "Efficient variational quantum simulator incorporating active error minimization", "Phys. Rev. X", "7", "021050", "2017", "10.1103/PhysRevX.7.021050")
add("kandala2019", "Kandala, A. et al.", "Error mitigation extends the computational reach of a noisy quantum processor", "Nature", "567", "491--495", "2019", "10.1038/s41586-019-1040-7")
add("kim2023", "Kim, Y. et al.", "Evidence for the utility of quantum computing before fault tolerance", "Nature", "618", "500--505", "2023", "10.1038/s41586-023-06096-3")
add("vandenberg2022", "van den Berg, E., Minev, Z. K. & Temme, K.", "Model-free readout-error mitigation for quantum expectation values", "Phys. Rev. A", "105", "032620", "2022", "10.1103/PhysRevA.105.032620")
add("viola1999", "Viola, L., Knill, E. & Lloyd, S.", "Dynamical decoupling of open quantum systems", "Phys. Rev. Lett.", "82", "2417--2421", "1999", "10.1103/PhysRevLett.82.2417")
add("wallman2016", "Wallman, J. J. & Emerson, J.", "Noise tailoring for scalable quantum computation via randomized compiling", "Phys. Rev. A", "94", "052325", "2016", "10.1103/PhysRevA.94.052325")
add("self2024", "Self, C. N., Benedetti, M. & Amaro, D.", "Protecting expressive circuits with a quantum error detection code", "Nat. Phys.", "20", "219--224", "2024", "10.1038/s41567-023-02282-2", "chk")
add("he2024", "He, Z. et al.", "Performance of quantum approximate optimization with quantum error detection", "Preprint at https://arxiv.org/abs/2409.12104", "", "", "2024", "", "chk")
add("ibm2026", "IBM Quantum.", "Our first Nighthawk QPU and latest Heron QPU are now available", "https://quantum.cloud.ibm.com/announcements/en/product-updates/2026-01-05-nighthawk", "", "", "2026", "")
add("qiskit2024", "Javadi-Abhari, A. et al.", "Quantum computing with Qiskit", "Preprint at https://arxiv.org/abs/2405.08810", "", "", "2024", "")
add("fujii2017", "Fujii, K. & Nakajima, K.", "Harnessing disordered-ensemble quantum dynamics for machine learning", "Phys. Rev. Appl.", "8", "024030", "2017", "10.1103/PhysRevApplied.8.024030")
add("mujal2023", "Mujal, P. et al.", "Time-series quantum reservoir computing with weak and projective measurements", "npj Quantum Inf.", "9", "16", "2023", "10.1038/s41534-023-00682-z", "chk")
add("ciccarello2022", "Ciccarello, F., Lorenzo, S., Giovannetti, V. & Palma, G. M.", "Quantum collision models: open system dynamics from repeated interactions", "Phys. Rep.", "954", "1--70", "2022", "10.1016/j.physrep.2022.01.001")
add("sannia2026", "Sannia, A., Rodr\\'iguez, R. R., Giorgi, G. L. & Zambrini, R.", "Non-Markovianity and memory enhancement in quantum reservoir computing", "npj Quantum Inf.", "12", "111", "2026", "10.1038/s41534-026-01257-4", "src")
add("jaeger2004", "Jaeger, H. & Haas, H.", "Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication", "Science", "304", "78--80", "2004", "10.1126/science.1091277")
add("zhang2026", "Zhang, C. et al.", "Experimental demonstration of quantum continual learning with superconducting qubits", "npj Quantum Inf.", "12", "28", "2026", "10.1038/s41534-025-01174-y", "src")


def nature(key, tex=True):
    r = dict(R[key]); r["authors"] = r["authors"].replace(" & ", " \\& ")
    j = r["journal"]
    if j.startswith("Preprint") or j.startswith("http"):
        s = f"{r['authors']} {r['title']}. {j if j.startswith('Preprint') else chr(92) + 'url{' + j + '}'} ({r['year']})."
    else:
        vol = f" \\textbf{{{r['vol']}}}," if r["vol"] else ""
        pg = f" {r['pages']}" if r["pages"] else ""
        doi = f" https://doi.org/{r['doi']}" if (r["doi"] and not r["vol"]) else ""
        s = f"{r['authors']} {r['title']}. \\textit{{{j}}}{vol}{pg} ({r['year']}).{doi}"
    return s


def bibtex(key):
    r = R[key]
    f = dict(author=r["authors"].replace(" et al.", " and others").replace(" & ", " and ").rstrip("."),
             title="{" + r["title"] + "}", journal=r["journal"], volume=r["vol"], pages=r["pages"], year=r["year"], doi=r["doi"])
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items() if v)
    return f"@article{{{key},\n{body}\n}}\n"
