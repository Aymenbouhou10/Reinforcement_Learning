import base
import numpy as np
env = base.Maze()

# Boucle d'actions-affichage
def executer_politique(env, pi):
    state = env.reset()
    fini = False
    while not fini:
        env.render()
        action = np.random.choice(4, p=pi[state])
        state, r, fini, _ = env.step(action)
        print("Etat", state, "- action", action, "- récompense", r)
    env.render()


def calculer_q(env, v, gamma):
    return np.sum(env.p() * 
            (env.r() + gamma*v[np.newaxis, np.newaxis, :]), axis=2)


def ipe(env, pi, gamma, epsilon):
    ns = env.get_nb_states()

    v = np.zeros((ns))
    q = calculer_q(env, v, gamma)
    nv = np.sum(pi * q, axis=1)
    
    delta = np.sum(np.abs(nv - v))
    while delta > epsilon:
        v = nv
        q = calculer_q(env, v, gamma)
        nv = np.sum(pi * q, axis=1)
        delta = np.sum(np.abs(nv - v))
    
    return nv    

def greedy(v, env, gamma):
    ns = env.get_nb_states()
    na = env.get_nb_actions()
    pi = np.zeros((ns, na))
    q = calculer_q(env, v, gamma)
    
    for s, qs in enumerate(q):
        best_action = np.argmax(qs)
        pi[s, best_action] = 1.
        
    return pi
    
def policy_iteration(env, gamma, epsilon):
    ns = env.get_nb_states()
    na = env.get_nb_actions()

    pi = np.ones((ns, na))/na
    v = ipe(env, pi, gamma, epsilon)
    npi = greedy(v, env, gamma)
    delta = np.sum(np.abs(npi - pi))
    while delta > epsilon:
        pi = npi
        v = ipe(env, pi, gamma, epsilon)
        npi = greedy(v, env, gamma)
        delta = np.sum(np.abs(npi - pi))
        
    return npi
    
def value_iteration(env, gamma, epsilon):
    ns = env.get_nb_states()

    v = np.zeros((ns))

    q = calculer_q(env, v, gamma)
    nv = np.max(q, axis=1)    
    delta = np.sum(np.abs(nv - v))
    while delta > epsilon:
        v = nv
        q = calculer_q(env, v, gamma)
        nv = np.max(q, axis=1)    
        delta = np.sum(np.abs(nv - v))
        
    return greedy(nv, env, gamma)