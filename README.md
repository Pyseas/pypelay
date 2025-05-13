*pypelay* is a Python package for the automation of S-lay stinger configuration and pipelay analysis using Orcaflex software.

## Todo:
- For radius >470m BR6 can't reach the pipe. Need to move tensioner_x forward by a few meters and redo all configs. Maybe 5m increments for full range.
- Check Orcaflex version
- Bugfix: Select radius not converging for big water depth


## Notes:
- When serving or building docs need to specify theme:

    ```
    ..\pypelay> mkdocs serve -t material
    ```

- How to install editable version (working folder):

    ```
    ..\working> uv init
    ..\working> uv add pandas
    ..\working> uv pip install -e "pypelay @ ../pypelay"
    ```
