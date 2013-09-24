/* transcript/src/getopt.c
 *
 * public domain getopt from mod.sources
 * RCSID: $Header: getopt.c,v 2.1 85/11/24 11:49:10 shore Rel $
 */

/*
**  This is a public domain version of getopt(3).
**  Bugs, fixes to:
**		Keith Bostic
**			ARPA: keith@seismo
**			UUCP: seismo!keith
**  Added NO_STDIO, opterr handling, Rich $alz (mirror!rs).
*/

#include <stdio.h>
#include <string.h>

/*
**  Error macro.  Maybe we want stdio, maybe we don't.
**  The (undocumented?) variable opterr tells us whether or not
**  to print errors.
*/

#ifdef	NO_STDIO

#define tell(s)								\
	if (opterr)							\
	{								\
	    char	ebuf[2];					\
	    (void)write(2, nargv, (unsigned int)strlen(nargv));		\
	    (void)write(2, s, (unsigned int)strlen(s));			\
	    ebuf[0] = optopt;						\
	    ebuf[1] = '\n';						\
	    (void)write(2, ebuf, 2);					\
	}

#else

#define tell(s)								\
	if (opterr)							\
	    (void)fputs(*nargv, stderr),				\
	    (void)fputs(s,stderr),					\
	    (void)fputc(optopt, stderr),				\
	    (void)fputc('\n', stderr)

#endif


/* Global variables. */
static char	 EMSG[] = "";
int		 opterr = 1;		/* undocumented error-suppressor*/
int		 optind = 1;		/* index into argv vector	*/
int		 optopt;		/* char checked for validity	*/
char		*optarg;		/* arg associated with option	*/


int
getopt(int nargc, char **nargv, char *ostr)
{
    static char		 *place = EMSG;	/* option letter processing	*/
    register char	 *oli;		/* option letter list index	*/

    if (!*place)			/* update scanning pointer	*/
    {
	if (optind >= nargc || *(place = nargv[optind]) != '-' || !*++place)
	    return(EOF);
	if (*place == '-')		/* found "--"			*/
	{
	    optind++;
	    return(EOF);
	}
    }
    /* option letter okay? */
    if ((optopt = *place++) == ':' || (oli = strchr(ostr, optopt)) == NULL)
    {
	if (!*place)
	    optind++;
	tell(": illegal option -- ");
	goto Bad;
    }
    if (*++oli != ':')			/* don't need argument		*/
    {
	optarg = NULL;
	if (!*place)
	    optind++;
    }
    else				/* need an argument		*/
    {
	if (*place)
	    optarg = place;		/* no white space		*/
	else
	    if (nargc <= ++optind)
	    {
		place = EMSG;
		tell(": option requires an argument -- ");
		goto Bad;
	    }
	    else
		optarg = nargv[optind];	/* white space			*/
	place = EMSG;
	optind++;
    }
    return(optopt);			/* dump back option letter	*/
Bad:
    return('?');
}
