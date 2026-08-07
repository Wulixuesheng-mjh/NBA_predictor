"""NBA Predictor ML module — lazy imports to avoid ORM double-registration.

When running in the trainer container (which copies backend ORM models alongside
ML code), importing all submodules eagerly would trigger SQLAlchemy model
registration from multiple code paths, causing "Table already defined" errors.

All public classes remain available — just import them directly from their
submodules rather than from the package root.
"""