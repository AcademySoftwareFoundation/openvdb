################################################################################
# Strip rpaths
.PHONY: strip-rpath-%
strip-rpath-%:
	@echo "stripping rpath from libs for version $*..."
	@cd $(STRIP_PATH)/$* && \
	find -name "*.so" -exec patchelf --set-rpath '' {} \;
