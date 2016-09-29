;; Set emacs to follow the OpenVDB's coding style.
;; (http://www.openvdb.org/documentation/doxygen/codingStyle.html)
;;
;; Disable tab, set the style to K&R and fixing the indentation width
;; and bracket placement.

( (nil . ( (indent-tabs-mode . nil)
	   (c-file-style . "k&r")
           (c-basic-offset . 4)
           (eval . (progn (c-set-offset 'inline-open '0)))
           )))
