# make_new_model
Modify a SACS base model according to the contents of a spreadsheet.

Possible modifications are:

- Update title and analysis type description
- Add new / modify existing joints
- Add new / modify existing members
- Remove existing / add new basic load cases
- Create load combinations
- Flood selected GRUPS (also inserts HYDRO and HYDRO2 lines)
- Insert LCSEL
- Insert notes (as comments)

The following sections describe the use of each of the sheets in the spreadsheet.

## TITLE
Input title and analysis type description in cells B1 and B2.

The script looks for a line in the base file containing just the word *TITLE* and inserts the title.

The script looks for a line in the base file containing *ANALYSIS TYPE* and replaces with:

~~~
****   ANALYSIS TYPE  : analysis_type
~~~

## joints
Adds new joints or modifies existing joints. Blank cells in the spreadsheet are ignored.

If a joint doesn't exist in the base file then a new member is created.

For PILEHD fixity enter PILEHD in the FX column.

## members
Adds new members or modifies existing members. Blank cells in the spreadsheet are ignored.

If a member doesn't exist in the base file then a new member is created.

## LOADCN
This sheet contains a list of basic load cases and a list of text file names.

### Basic load cases
Each load case is either kept or removed based on the contents of column C.

A list of load cases (including descriptions) is inserted as a comment block immediately after the line containing *BASIC LOAD CASES* in the following format (e.g.):

~~~
****   BASIC LOAD CASES                                           *
****   LOAD CASE:  0010 = MODELLED DEAD LOADS                     *
~~~

### Text files
For each text file the script looks for the file in a subfolder named *load_files*. The contents of all files are combined and inserted into the base file at the location of the following line:

~~~
***ADD LOADS
~~~

Note that the load cases in the text files are not added to the BASIC LOAD CASES comment section.

## LCOMB
The script inserts a new LCOMB line for each column in this sheet, starting at column D.

Load combinations are inserted immediately before the first *END* in the base file.

### TODO

- Compare load cases in spreadsheet with load cases in base file, alert any differences.
- Get list of load cases from text files and add to BASIC LOAD CASES comment section.

## Notes
This sheet contains a list of notes, starting at cell A2. Notes are inserted in the base file at the location of the following line:

~~~
***ADD NOTES
~~~

## LCSEL
Adds LCSEL line(s) to the base file, based on contents of column B.

LCSEL does not exist in the base file. It's inserted immediately after CODE line(s).

## FLOOD
Flood does 2 things:

- Updates LDOPT line with water depth and *HYD* option
- Inserts the following lines just before *UCPART*.

~~~
HYDRO +ZISEXTFLNO  I20.000              2.000     1.025     1.000     0.250
HYDRO2    0.900IN0.8002.000
~~~
