/*
 * Copyright (c) 2019 Szabolcs Horvát.
 *
 * See the file LICENSE.txt for copying permission.
 */

// These #includes are redundant. They are only for the IDE.
#include "LTemplate.h"
#include "LTemplateHelpers.h"

#include <csetjmp>
#include <cstdio>

namespace mma {


WolframLibraryData libData;

namespace detail { // private

    int MBuffer::sync() {
        // If the last character is a newline, remove it.
        // This makes it convenient to flush with std::endl
        if (pptr() > pbase() && pptr()[-1] == '\n')
            pbump(-1);

        *pptr() = '\0';
        std::ptrdiff_t n = pptr() - pbase();
        if (n > 0)
            mma::print(&buf.front());
        pbump(-n);
        return 0;
    }

    std::streambuf::int_type MBuffer::overflow(std::streambuf::int_type ch) {
        if (ch != traits_type::eof()) {
            massert(pptr() == epptr()); // overflow should only be called if the buffer is out of space

            *pptr() = traits_type::to_char_type(ch);

            std::size_t offset = pptr() - pbase();
            std::size_t old_buf_size = buf.size() - 1;
            buf.resize( 2*old_buf_size + 1 );

            setp( &buf.front(), &buf.back() );
            pbump( offset + 1 );
        }
        return ch;
    }

    static MBuffer mbuf;

    static std::jmp_buf jmpbuf;

} // end namespace detail


std::ostream mout(&detail::mbuf);


// It would be natural for fatal_error() to take a messae argument that will be printed before returning.
// However, that would create an incentive to construct a string object that would need to be freed.
// longjmp() prevents calls to desctructors, thus freeing the string would be complicated.
void fatal_error() {
    std::longjmp(detail::jmpbuf, 1);
}


void message(const char *msg, MessageType type) {
    if (msg == NULL)
        return;

    if (libData->AbortQ())
        return; // trying to use the MathLink connection during an abort will break it

    const char *tag;
    switch (type) {
    case M_ERROR:
        tag = "error";
        break;
    case M_WARNING:
        tag = "warning";
        break;
    case M_ASSERT:
        tag = "assert";
        break;
    case M_INFO:
    default:
        tag = "info";
    }

    MLINK link = libData->getMathLink(libData);
    MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Message", 2);
            MLPutFunction(link, "MessageName", 2);
                MLPutSymbol(link, LTEMPLATE_MESSAGE_SYMBOL);
                MLPutString(link, tag);
            MLPutString(link, msg);
    libData->processMathLink(link);
    int pkt = MLNextPacket(link);
    if (pkt == RETURNPKT)
        MLNewPacket(link);
}


} // end namespace mma
