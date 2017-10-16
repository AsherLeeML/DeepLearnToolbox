function dbn = dbnsetup(dbn, inputSize, opts)
    dbn.sizes = [inputSize, dbn.sizes];

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = randn(dbn.sizes(u + 1), dbn.sizes(u))/sqrt(dbn.sizes(u + 1));
        dbn.rbm{u}.vW = randn(dbn.sizes(u + 1), dbn.sizes(u))/sqrt(dbn.sizes(u + 1));

        dbn.rbm{u}.b  = randn(dbn.sizes(u), 1)/sqrt(dbn.sizes(u));
        dbn.rbm{u}.vb = randn(dbn.sizes(u), 1)/sqrt(dbn.sizes(u));

        dbn.rbm{u}.c  = randn(dbn.sizes(u + 1), 1)/sqrt(dbn.sizes(u + 1));
        dbn.rbm{u}.vc = randn(dbn.sizes(u + 1), 1)/sqrt(dbn.sizes(u + 1));
    end

end
