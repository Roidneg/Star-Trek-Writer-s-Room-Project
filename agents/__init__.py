"""Project specialist agents.

Keep this a regular local package so an unrelated installed ``agents``
package cannot replace the project's namespace during ordinary imports.
Agent modules are loaded explicitly; importing this package starts no models.
"""
