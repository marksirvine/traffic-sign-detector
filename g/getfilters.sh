rm -r filters
scp -oProxyCommand="ssh -W %h:%p gc14768@snowy.cs.bris.ac.uk" -r gc14768@bc4login.acrc.bris.ac.uk:cw/filters ./