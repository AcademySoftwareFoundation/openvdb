// SPDX-License-Identifier: MIT
/**

Doxygen Awesome
https://github.com/jothepro/doxygen-awesome-css

Copyright (c) 2022 - 2025 jothepro

*/

class DoxygenAwesomeParagraphLink {
    // Icon from https://fonts.google.com/icons
    // Licensed under the Apache 2.0 license:
    // https://www.apache.org/licenses/LICENSE-2.0.html
    static icon = `<svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 0 24 24" width="20px"><path d="M0 0h24v24H0V0z" fill="none"/><path d="M17 7h-4v2h4c1.65 0 3 1.35 3 3s-1.35 3-3 3h-4v2h4c2.76 0 5-2.24 5-5s-2.24-5-5-5zm-6 8H7c-1.65 0-3-1.35-3-3s1.35-3 3-3h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-2zm-3-4h8v2H8z"/></svg>`
    static title = "Permanent Link"
    static init() {
        $(function() {
            $(document).ready(function() {
                document.querySelectorAll(".contents a.anchor[id], .contents .groupheader > a[id]").forEach((node) => {
                    let anchorlink = document.createElement("a")
                    anchorlink.setAttribute("href", `#${node.getAttribute("id")}`)
                    anchorlink.setAttribute("title", DoxygenAwesomeParagraphLink.title)
                    anchorlink.classList.add("anchorlink")
                    node.classList.add("anchor")
                    anchorlink.innerHTML = DoxygenAwesomeParagraphLink.icon
                    node.parentElement.appendChild(anchorlink)
                })
            })
        })
    }
}
