# Photon decays

# Implementation
In the present Jupyternootbook we implement the equation derived by Dr. Mazeliauskas to compute the feed-down generated from parent particle a to a particle b due to the decay channel a-> bc. 

The first cell defines all relevant variables with some built-in error-handling for forbidden cases (ma = 0) and cases used for debugging (complex square roots and similar), which print out debugging messages that can be turned off with the bool Debug. All of these functions' names start with "get" and end with the variables they compute.

In particular, we have implemented the limit case where the child particle, b, is massless, which in particular creates three cases for the lower-integration bound qTminus - mb !=0 ,  mb = 0 && mc !=0 a and mb == 0 && mc ==0 - and  two for the upper integration case qTplus - mb ==0 and mb !==0.

The integrands for the massive case and the massless case are at the end of the cell and are integrand_factor_massive and integrand_factor_massless. For the massive case we have we have also considered the case where the variable a_{-} becomes negative and change the \phi-integrand accordingly. 

The second cell defined a standard MC-integration routine which uses gsl_monte_vegas. The third one defines an Integrator function which takes a function, integration limits, and desired relative error. This function uses quad standard integration routine, and if the relative error is not achieved, it switches to the more sophisticated MC-integration routine. If this integration strategy is to be used, we recommend to confirm that vegas package is installed first. Or install it with:
    pip install vegas
While this integration method can in general achieve a greater presition, our test show that, as it is currently implemented, it is too slow. 

Finally, we define a function which takes the analytical expression of a spectrum and computes the feed-down to the decay particle b - getFeedDown_anadNa. As parameters, this function takes a np.array with the p_T to be computed, doubles for the particles' masses, a function dNa_dpT defining the spectrum of the parent particle and its arguments which have to be given as a tuple or similar objects. The function allows to set the relative error EpsRel, default value 1e-4, and to swtich off the debugging outputs with bool Debug, which is by default False. In addition to this function, we defined getFeedDown_anadNa_safe with the same arguments, but using the Integrator function explained above.

-- Note: The defined functions to compute all relevant terms can be easily used to implement a function that takes the spectrum from a .txt file or .csv instead of an analytical expression. For the moment, this is not done. 

# Tests

Both massive and massless cases have been tested using a thermal spectrum for the parent particle. No major issue has been encountered in the code; however, the massive case shows a p_T range where the implemented analytical formulas return NaN values. The massless case shows this behaviour only around q_T = 0 and p_T = 0. 

For the massless case, we computed the feed-down to photon spectrum coming from \pi^0\to \gamma \gamma at T = 230 MeV. For this case, we also computed the ratio \gamma_{decay}/\pi^0 to allow comparison with experimental data. A rough comparison shows no major deviation from the computed ratio to the measured one. 

For the massive case, we computed the feed-down to \rho^+ \to \pi^+ \pi^0 at the same temperature. 

## Particle conservation
pending.

# Extra plots

After the testing section, we have plotted the behaviour of different variables as a function of q_T or p_T to study them. In particular, at the end, we show a plot of the integrand for mb = 0 and mb!=0  as a function of p_T and q_T. 