# Fly Language Study

### Mapping language difficulty by tracing the neural wiring of fruit flies

## What do we explore?

- **Learning speed** — Which language reaches the same performance level faster?
- **Learning order** — Does the first language learned help or hinder the next?
- **Memory and forgetting** — How much of a new language is forgotten after learning it, and how much can be recovered through review?

**The study compares three languages — English, German, and Korean — across six learning orders.**

Using a computer model built from the actual neural connectivity map (connectome) of a fruit fly’s brain, we investigate differences in language learning speed and learning order effects.

**5,000 neurons • 524,324 connections • compared in the same model**

https://github.com/user-attachments/assets/4acea28c-f871-4d88-9dc4-52d10b3214bc


The model decides whether two sentences have the same meaning. We compare four tasks: role relations, negation, spatial relations, and quantity — across:

single-language learning
sequential learning in six orders
mixed-language learning across three languages

## So far

n the initial exploration of the first two random seeds, the order in which Korean was learned first was the fastest. We also observed a pattern in which performance in the previously learned language temporarily dropped and then recovered through review. We are currently running repeated pilot studies to determine the scale of the final experiment, and the final conclusion has not yet been finalized.

[View exploratory results →](reports/EXPLORATORY_V5_WORDBOUND_ORDER.md)

## Learn more

[Study design](docs/PROTOCOL_V5.md) · [Progress log](reports/MAIN_STUDY_PLAN_V5.md) · [Code](src/flystudy) · [background](docs/SOURCE_AUDIT.md)

---

This is a research project exploring whether there are general trends in language learning difficulty and order effects. As a first step, we measure learning efficiency in this model, task setup, and input representation, and the broader generality will be tested in follow-up studies.
