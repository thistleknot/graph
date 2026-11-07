Global aggregate pipeline
=========================

Scope
-----

Corpus-wide counting questions are answered from a complete term-count index,
built once per pinned section build, and from durable jobs that classify a
sample of papers with a language model. A job's plan is fixed when it is created,
each provider call is recorded before and after it runs, and a completed call is
never repeated. Interrupted jobs are reported incomplete and are resumed only
for calls that did not complete. The sources are ``src/global_aggregate.py``
and ``src/aggregate_jobs.py``.

Sphinx-Needs components and workflow
------------------------------------

.. comp:: Term-count builder
   :id: C_GA_COUNTS
   :reads: C_SECTION_DB
   :persists: C_GA_STORE

   Groups the sections of the pinned build into papers, assigns each paper to its
   main community, and counts each term once per paper in each community.

.. comp:: Global count store
   :id: C_GA_STORE

   Holds the paper assignments, community sizes, term counts and routing
   signatures of one complete index. The index is marked complete last.

.. comp:: Job executor
   :id: C_GA_JOBS
   :persists: C_GA_JOBFILE

   Runs accepted jobs within a bounded number of concurrent jobs and records each
   provider call before it is made and its reply when it arrives.

.. comp:: Job file
   :id: C_GA_JOBFILE

   Local file holding each job's fixed plan, its progress, its result and the
   record of each provider call.

.. comp:: Paper classifier
   :id: C_GA_CLASSIFIER
   :calls_model: C_MODEL
   :persists: C_GA_JOBFILE

   Asks the model whether each sampled paper answers yes, no or unsure, and counts
   only replies in that form.

.. comp:: Job interface
   :id: C_GA_API
   :requests: C_GA_JOBS

   Submits a job, reports its state, and resumes an interrupted job when the
   chat filter polls it.

Job handoffs
------------

.. needflow:: Global aggregate handoffs
   :filter: id.startswith("C_GA_")
   :link_types: reads, persists, requests, calls_model
   :show_link_names:
   :direction: right
   :alt: Global aggregate pipeline from the term-count builder through the count store, job interface, job executor, job file and paper classifier calling the model
